# 使用vLLM + Ray做offline batch inference

目前我所知道最优雅的LLM batch inference方案，避免手动shard dataset、多机调度。救赎之道，就在其中。

### 镜像

`vllm:v0.6.1_ray_torch2.5.1_cuda11.8.0_python3.10.12`

### 使用

使用以上镜像按需启动多台机器，在launcher上运行`bash init_ray_cluster.sh`启动ray机群。然后参考`examples/infer/offline_batch_inference.py`编写自己的infer脚本，最后`python3 offline_batch_inference.py`即可