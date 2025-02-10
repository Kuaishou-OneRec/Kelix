#!/bin/bash

set -x

ckpt_path=$1
deploy_path=$2
model_tag=$3
gpu_id=$4
port=$5

local_ip=`ip a show eth01 | grep inet | grep -v inet6 | awk '{print $2}' | awk -F '/' '{print $1}'`

if [ -n "$1" ] && [ -n "$2" ] && [ -n "$3" ] && [ -n "$4" ] && [ -n "$5" ]; then
    echo "ckpt_path=${ckpt_path}"
    echo "deploy_path=${deploy_path}"
    echo "model_tag=${model_tag}"
    echo "service on GPU${gpu_id}"
    echo "http://${local_ip}:${port}"
else
    echo "error args!!!"
    echo "example: bash run_vllm.sh ckpt_path deploy_path model_tag gpu_id port"
    exit -1
fi

echo "step1: deploy model"
model_output_path=${deploy_path}/${model_tag}
if [ -e ${model_output_path} ]; then
    echo "error!!!"
    echo "${model_output_path} path exist!"
    exit -1
else
    if [ -e ${ckpt_path}/mp_rank_00_model_states.pt ]; then
        mkdir -p ${model_output_path}
        python3 trans_model.py --input_dir=${ckpt_path} --output_dir=${model_output_path} 
    else
        echo "error!!!"
        echo "${ckpt_path}/mp_rank_00_model_states.pt not found"
        exit -1
    fi
fi

echo "step2: start vllm service"
if [ -e ${deploy_path}/${model_tag}/pytorch_model.bin ]; then
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    cd ${deploy_path}
    mkdir -p ./logs
    nohup vllm serve ${model_tag} --dtype auto --port ${port} --api-key token-123456 > ./logs/${model_tag}.log 2>&1 &
else
    echo "error!!!"
    echo "${deploy_path}/${model_tag}/pytorch_model.bin not found"
    exit -1
fi

echo "server log: ${deploy_path}/logs/${model_tag}.log"
echo "api url: http://${local_ip}:${port}/v1"
