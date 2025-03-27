#!/bin/bash

set -x

ckpt_path=$1
deploy_path=$2

if [ -n "$1" ] && [ -n "$2" ]; then
    echo "ckpt_path=${ckpt_path}"
    echo "deploy_path=${deploy_path}"
else
    echo "error args!!!"
    echo "example: bash run_vllm.sh ckpt_path deploy_path"
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