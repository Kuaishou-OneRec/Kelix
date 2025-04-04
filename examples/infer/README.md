# 离线Inference

## 方式1: 使用vLLM + Ray

### 镜像

`deepspeed:v1.6_vllm0.8.1_ray_torch2.5.1_cuda11.8.0_python3.10.12_transformers4.49_zh`

### 使用

原理：通过Ray起多个vLLM Engine，每个Engine对应一个Rank，对数据集按照Rank做Sharding，每个Rank调用自己的Engine处理对应的Shard，从而实现多机多卡分布式离线推理。

所以，首先需要制作数据集，目前只支持Parquet格式。数据集格式参考项目主页README的Data Preparation部分。数据集文件个数需要大于等于N_ENGINES(vLLM Engine个数) * NUM_WORKERS(torch dataset的worker个数)。vLLM Engine个数取决于模型推理时的tp_size和机器数量，N_GPUS // TP_SIZE = N_ENGINES。通常Qwen2-VL 7B推荐tp=4，72B推荐tp=8。

数据集制作好之后，参考本文件夹下的`run_vllm_inference.sh`，自己修改对应参数，batch_size可以设大一些，vLLM是串行处理的，不会OOM。

## 方式2：SGlang

原生torch，不依赖ray做通信，但目前社区没有vLLM活跃。

### 镜像
镜像：[sglang:v0.4.0_cuda11.8.0_python3.10_v1](https://kml.corp.kuaishou.com/v2/#/project/10079/image/dockerfile/450103)

### 使用

运行`bash run_sglang_multi.sh model_path tp_size`即可。

以DeepSeek-R1为例，需要2xH800，运行`bash run_sglang_distributed.sh /llm_reco_ssd/zhouyang12/models/DeepSeek-R1 16`，从启动到载入模型大概需要20分钟；之后在rank0节点可以直接通过http请求访问，兼容OpenAI接口。使用结束之后可以关掉任务。

```
curl -s http://localhost:30000/v1/chat/completions   -d '{"model": "/llm_reco_ssd/zhouyang12/models/DeepSeek-R1/", "messages": [{"role": "user", "content": "已知锐角的定义是小于90度，钝角是大于90度，开水是100度，所以开水是钝角?"}]}'
```


