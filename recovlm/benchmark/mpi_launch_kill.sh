#!bin/bash
unset http_proxy https_proxy

hostfile=/etc/mpi/hostfile
gpus=$(head -n 1 /etc/mpi/hostfile | grep -Eo '[0-9]+$')
slots=8
sed -i "s/slots=[0-9]*/slots=$slots/" $hostfile

hostfile=/etc/mpi/hostfile
gpus=$(head -n 1 /etc/mpi/hostfile | grep -Eo '[0-9]+$')
slots=1
sed -i "s/slots=[0-9]*/slots=$slots/" $hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')
all=$(grep -n  "" /etc/mpi/hostfile | wc -l)
t=$(($all*$gpus))
set -x
rank=$GLOBAL_ID
echo "GPU:"
echo $gpus
echo "slots:"
echo $all
echo "Rank:"
echo $rank
echo "t:"
echo $t

ip=$LOCAL_IP
echo $ip
Port1=20280

mpirun --allow-run-as-root \
    -hostfile /etc/mpi/hostfile \
    --bind-to socket \
    --map-by socket \
    -x NCCL_SOCKET_IFNAME=eth01 \
    -x NCCL_DEBUG=WARN \
    -x NCCL_NET_GDR_LEVEL=3 \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_SOCKET_NTHREADS=4 \
    -x NCCL_NSOCKS_PERTHREAD=4 \
    -x NCCL_IB_DISABLE=0 \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
	sh -x run_kill.sh

