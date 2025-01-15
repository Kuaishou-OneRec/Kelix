ps -aux|grep ray_batch_infer.py|grep -v grep|awk '{print $2}'|xargs kill -9
ps -aux|grep run_ray|grep -v grep|awk '{print $2}'|xargs kill -9
