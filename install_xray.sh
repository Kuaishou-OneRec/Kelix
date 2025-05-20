#!/bin/bash

function print_green() {
    echo -e "\e[32m\e[100m$1\e[0m"
}

function print_red() {
    echo -e "\e[31m\e[100m$1\e[0m"
}

export http_proxy=http://oversea-squid4.sgp.txyun:11080
export https_proxy=http://oversea-squid4.sgp.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

_os_=$(grep -Ei '^ID=' /etc/os-release | cut -d'=' -f2 | tr -d '"')
if [ "${_os_}" == "ubuntu" ]; then
    _pkg_="deb"
    dpkg --configure -a &> /dev/null || true
    apt update && apt-get install -y net-tools iproute2 lldpd bind9-utils ethtool iputils-ping
    if out=$(xray update) && grep -q "update complete" <<< "$out" ; then print_green "Info: xray auto update success" ; exit 0 ; fi
    if xray --help &> /dev/null ; then
        print_green "removing old version of xray"
        apt-get remove -y xray &> /dev/null || true
    fi
elif [ "${_os_}" == "centos" ]; then
    _pkg_="rpm"
    yum install -y --nogpgcheck net-tools iproute lldpd bind-utils ethtool iputils --skip-broken
    if out=$(xray update) && grep -q "update complete" <<< "$out" ; then print_green "Info: xray auto update success" ; exit 0 ; fi
    if xray --help &> /dev/null ; then
        print_green "removing old version of xray"
        yum remove -y xray &> /dev/null || true
    fi
else print_red "Error: unsupported os for xray"; exit 1
fi

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

_name_=xray_latest.${_gpu_}_amd64.${_pkg_}
_ip_=$(hostname -i)
print_green "Info: downloading ${_name_}"
wget -O "${_ip_}${_name_}" -q https://halo.corp.kuaishou.com/api/cloud-storage/v1/public-objects/xray/${_name_}
PATH=$(sed -e 's/:\/opt\/xray\/deps//g' -e 's/^\/opt\/xray\/deps://' <<< "$PATH")  # 排除上次安装卸载的影响
which mpirun > /etc/real_mpirun_path_used_by_xray
flag=0
if [ "${_os_}" == "ubuntu" ]; then
    dpkg --configure -a &> /dev/null || true
    if apt-get install -y ./"${_ip_}${_name_}"; then
        print_green "Info: xray install success"
    else print_red "Error: xray install failed"; flag=3
    fi
elif [ "${_os_}" == "centos" ]; then
    if yum localinstall -y ./"${_ip_}${_name_}"; then
        print_green "Info: xray install success"
    else print_red "Error: xray install failed"; flag=3
    fi
else print_red "Error: unsupported os for xray"; flag=1
fi
rm "${_ip_}${_name_}" &> /dev/null || true
exit "$flag"
