#!/bin/bash

set -x

# 在此处设置固定路径
ckpt_path="/path/to/your/checkpoint"
deploy_path="/path/to/deploy/directory"
model_tag="your_model_tag"  # 添加model_tag变量的定义

echo "ckpt_path=${ckpt_path}"
echo "deploy_path=${deploy_path}"

echo "step1: deploy model"
model_output_path=${deploy_path}/${model_tag}
if [ -e ${model_output_path} ]; then
    echo "错误!!!"
    echo "${model_output_path} 路径已存在!"
    exit -1
else
    if [ -e ${ckpt_path}/mp_rank_00_model_states.pt ]; then
        mkdir -p ${model_output_path}
        python3 trans_model.py --input_dir=${ckpt_path} --output_dir=${model_output_path} 
    else
        echo "错误!!!"
        echo "${ckpt_path}/mp_rank_00_model_states.pt 文件不存在"
        exit -1
    fi
fi 