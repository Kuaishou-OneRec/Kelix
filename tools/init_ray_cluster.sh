script_dir=$(cd $(dirname $0); pwd)
head_node_addr=$(awk 'NR==1 {print $1}' /etc/mpi/hostfile)
port=6381
echo "script dir: $script_dir"
echo "head_node_addr: $head_node_addr"

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "bash $script_dir/init_ray.sh $head_node_addr $port"