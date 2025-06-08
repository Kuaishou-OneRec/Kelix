#!/bin/bash
# bash install.sh  # install on current host
# bash install.sh all  # install on all hosts by mpirun


function print_green() { echo -e "\e[32m\e[100m$1\e[0m" ; }
function print_red() { echo -e "\e[31m\e[100m$1\e[0m" ; }
export http_proxy=http://oversea-squid4.sgp.txyun:11080
export https_proxy=http://oversea-squid4.sgp.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
cloud_storage="https://halo.corp.kuaishou.com/api/cloud-storage/v1/public-objects"


# Step0: 多机安装
install_dir=$1
if [ "$install_dir" == "all" ]; then
    ceph_df_info=$(xargs -I{} df -B 1G {} < <(awk '{print $3}' < <(grep "type ceph" < <(mount))))
    read -r ceph_dir ceph_size _ < <(tail -1 < <(sort -k2 -n < <(awk '!/^File/ {print $6,$4}' <<< "$ceph_df_info")))
    if [ ! -d "$ceph_dir" ] || [[ ! "$ceph_size" =~ ^[1-9][0-9]*$ ]]; then
        print_red "cannot find any vacant share directory on ceph to help install xray on all workers"; exit 2
    fi
    set -x
    env -i PATH="$(sed -e 's/^\/opt\/xray\/deps://' <<< "$PATH")" HOME="$HOME" mpirun --allow-run-as-root -pernode -hostfile /etc/mpi/mpi-hostfile \
        -x PATH bash -c "bash install_xray.sh ${ceph_dir} || echo -e '\e[31m\e[100m xray install failed on $(hostname) \e[0m'"
    exit 0
fi


# Step1: 检查OS版本，安装依赖包
_os_=$(grep -Ei '^ID=' /etc/os-release | cut -d'=' -f2 | tr -d '"')
if [ "${_os_}" == "ubuntu" ]; then
    _pkg_="deb"
    dpkg --configure -a &> /dev/null || true
    print_green "updating dependencies by apt"
    apt update &> /dev/null && apt-get install -y net-tools iproute2 lldpd bind9-utils ethtool iputils-ping &> /dev/null
elif [ "${_os_}" == "centos" ]; then
    _pkg_="rpm"
    print_green "updating dependencies by yum"
    yum install -y --nogpgcheck net-tools iproute lldpd bind-utils ethtool iputils --skip-broken &> /dev/null
else print_red "Error: unsupported os for xray, only centos and ubuntu are available"; exit 1
fi


# Step2: 检查GPU版本
if command -v nvcc &> /dev/null; then
    _gpu_="cuda"$(nvcc --version | grep release | sed 's/.*release //' | sed 's/\,.*//' | sed 's/\..*//')
elif nvcc=$(timeout 5s find /usr/ -type f -name nvcc -executable -print -quit) && [ -n "$nvcc" ]; then
    _gpu_="cuda"$("$nvcc" --version | grep release | sed 's/.*release //' | sed 's/\,.*//' | sed 's/\..*//')
elif command -v nvidia-smi &> /dev/null; then
    _gpu_="cuda"$(nvidia-smi | grep 'CUDA Version' | awk -F'CUDA Version: ' '{print $2}' | cut -d'.' -f1)
elif command -v amd-sli &> /dev/null || command -v rocm-sli &> /dev/null ; then
    _gpu_="rocm6"  # TODO: 动态识别
else print_red "Error: unrecognized cuda/rocm version"; exit 2
fi


# Step3: 更新xray
installed="0"
if out=$(xray update) && grep -q "update complete" <<< "$out" ; then
    print_green "Info: xray auto update success"
    installed="1"
elif xray --help &> /dev/null ; then
    print_green "Info: removing old version of xray"
    if [ "${_os_}" == "ubuntu" ]; then
        apt-get remove -y xray &> /dev/null || true
    else
        yum remove -y xray &> /dev/null || true
    fi
fi


# Step4: 安装xray
if [ "$installed" == "0" ]; then
    _name_=xray_latest.${_gpu_}_amd64.${_pkg_}
    _ip_=$(hostname -i)
    print_green "Info: downloading ${_name_}"
    wget -O "${_ip_}_${_name_}" -q "${cloud_storage}/xray/${_name_}"
    PATH=$(sed -e 's/:\/opt\/xray\/deps//g' -e 's/^\/opt\/xray\/deps://' <<< "$PATH")  # 排除上次安装卸载的影响
    which mpirun > /etc/real_mpirun_path_used_by_xray
    error="0"
    if [ "${_os_}" == "ubuntu" ]; then
        dpkg --configure -a &> /dev/null || true
        if apt-get install -y ./"${_ip_}_${_name_}"; then
            print_green "Info: xray install success"
        else print_red "Error: xray install failed"; error=3
        fi
    else
        if yum localinstall -y ./"${_ip_}_${_name_}"; then
            print_green "Info: xray install success"
        else print_red "Error: xray install failed"; error=4
        fi
    fi
    rm "${_ip_}_${_name_}" &> /dev/null || true
    if [ "$error" != "0" ]; then exit "$error"; fi
fi


# Step5: 下载合适的NCCL库
mount_info=$(mount); rank="$OMPI_COMM_WORLD_RANK"
if [ ! -d "$install_dir" ]; then install_dir="/opt/xray"; fi
if grep -q "$install_dir type ceph" <<< "$mount_info" && [[ $rank =~ ^[1-9][0-9]*$ ]]; then
    echo "only the host with rank=0 need to download nccl into shared directory"; exit 0
fi
KCCL_PATH="$install_dir""/kccl/${_os_}/${_gpu_}"
mkdir -p "$KCCL_PATH"; cd "$KCCL_PATH" || exit 1
remote_url="${cloud_storage}/user-cloud-storage/nccl-kai%2Fkccl%2F${_os_}-${_gpu_}%2F"
if latest=$(wget -qO - "${remote_url}latest"); then  # TODO 打点版本集成版本
    echo "downloading latest nccl $latest into ${KCCL_PATH}"
    wget -nc -qO "$latest" "${remote_url}$latest"  # TODO 建议不做覆盖更新，避免多任务对share目录冲突读写
    if [ -f "$latest" ]; then
        echo "$KCCL_PATH""/$latest" > /etc/real_kccl_used_by_xray
        print_green "Info: setting $(cat /etc/real_kccl_used_by_xray) as default nccl"
    else
        print_red "Error: failed to download nccl library ${remote_url}$latest"; exit 5
    fi
else
    print_red "Error: cannot check latest nccl in ${remote_url}"; exit 6
fi
