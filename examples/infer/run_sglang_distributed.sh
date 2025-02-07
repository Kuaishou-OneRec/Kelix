SCRIPT_DIR=$(cd $(dirname $0); pwd)
MASTER_ADDR=$(awk 'NR==1 {print $1}' /etc/mpi/hostfile)
PORT=20000
echo "script dir: $SCRIPT_DIR"
echo "master addr: $MASTER_ADDR:$PORT"

MODEL=$1
TP=$2

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "bash $SCRIPT_DIR/run_sglang.sh $MASTER_ADDR:$PORT $MODEL $TP"
