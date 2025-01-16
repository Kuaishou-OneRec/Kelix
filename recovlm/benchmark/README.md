镜像：registry.corp.kuaishou.com/kml-supercomputing-project/vllm:v0.6.1_ray_vlm_flash_attn_torch2.5.1_cuda11.8.0_python3.10.12
第一步sh -x run_config.sh配置好环境

第二步bash init_ray_cluster.sh 启动ray

第三步修改run_ray_while.sh中的model为待评估model folder，sh -x run_ray_while.sh即可评估当前model的所有checkpoint

失败后可以用 mpi_launch_kill.sh 杀掉进程

