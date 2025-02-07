### SGLang 多机部署

镜像：registry.corp.kuaishou.com/kml-supercomputing-project/sglang:v0.4.0_router_mpi4py_cuda11.8.0_python3.10

以下命令都只需要在RAN0执行：

1. 启动多个SGLang Runtime，实例个数=NGPUS / tp，`bash run_server.sh --model-path xxx --tp 8 --port 30000`，可以传入其他参数，参考https://docs.sglang.ai/backend/server_arguments.html

2. 等server启动后，启动SGLang Router，用于分发请求，`bash run_router.sh 8 50000 30000`，参数分别是tp、server_port、router_port，tp和上面保持一样，server_port和router_port不能相同，client通过router_port访问，由router进行负载均衡

3. 访问，参考batch_infer.py，可以多进程访问提高并行度。

4. 常见模型的启动

```shell
# Qwen2-VL-72B-Instruct
bash run_server.sh --model-path /llm_reco_ssd/zhouyang12/models/Qwen2-VL-72B-Instruct --tp 8 --port 30000

# Qwen2-VL-72B-Instruct-GPTQ-Int4
# 量化版本只需要4张卡，一台机器可以启动两个实例
bash run_server.sh --model-path /llm_reco_ssd/zhouyang12/models/Qwen2-VL-72B-Instruct-GPTQ-Int4 --tp 4 --port 30000

# DeepSeek-R1
# R1需要至少两台H800启动一个实例
bash run_server.sh --model-path /llm_reco_ssd/zhouyang12/models/Qwen2-VL-72B-Instruct-GPTQ-Int4 --tp 16 --port 30000
```