# 离线Inference

## 方式1: 使用vLLM + Ray做offline batch inference

~~目前我所知道最优雅的LLM batch inference方案，避免手动shard dataset、多机调度。救赎之道，就在其中。~~ 已经不是了，ray有点重，有坑不太好解决，后面可以使用MPI

### 镜像

`vllm:v0.6.1_ray_torch2.5.1_cuda11.8.0_python3.10.12`

### 使用

使用以上镜像按需启动多台机器，在launcher上运行`bash init_ray_cluster.sh`启动ray机群。然后参考`examples/infer/offline_batch_inference.py`编写自己的infer脚本，最后`python3 offline_batch_inference.py`即可

## 方式2：SGlang

原生torch，不依赖ray做通信，但目前社区没有vLLM活跃，如果需要跨节点部署，可以优先考虑SGLang。

### 镜像
镜像：[sglang:v0.4.0_cuda11.8.0_python3.10_v1](https://kml.corp.kuaishou.com/v2/#/project/10079/image/dockerfile/450103)

### 使用

运行`bash run_sglang_multi.sh model_path tp_size`即可。

以DeepSeek-R1为例，需要2xH800，运行`bash run_sglang_distributed.sh /llm_reco_ssd/zhouyang12/models/DeepSeek-R1 16`，从启动到载入模型大概需要20分钟；之后在rank0节点可以直接通过http请求访问，兼容OpenAI接口。使用结束之后可以关掉任务。

```
curl -s http://localhost:30000/v1/chat/completions   -d '{"model": "/llm_reco_ssd/zhouyang12/models/DeepSeek-R1/", "messages": [{"role": "user", "content": "已知锐角的定义是小于90度，钝角是大于90度，开水是100度，所以开水是钝角?"}]}'
```

### TODO
支持Offline Batch Inference

