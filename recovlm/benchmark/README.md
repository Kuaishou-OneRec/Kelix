第一步sh -x run_config.sh配置好环境

第二步bash init_ray_cluster.sh 启动ray

第三步修改run_ray_while.sh中的model为待评估model folder，sh -x run_ray_while.sh即可评估当前model的所有checkpoint

