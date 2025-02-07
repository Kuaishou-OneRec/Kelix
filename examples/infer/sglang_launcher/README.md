### SGLang 多机启动

1. 启动多个SGLang Runtime，实例个数=NGPUS / tp，`bash run_server.sh --model-path xxx --tp 8 --port 30000`，可以传入其他参数，参考https://docs.sglang.ai/backend/server_arguments.html

2. 启动SGLang Router，用于分发请求，`bash run_router.sh 8 50000 30000`，参数分别是tp、router_port、server_port

3. 访问，参考batch_infer.py，可以多进程访问提高并行度。