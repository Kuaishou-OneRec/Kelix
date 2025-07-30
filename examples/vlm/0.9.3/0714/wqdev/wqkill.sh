mpirun --allow-run-as-root --hostfile $1 --pernode bash -c  "pkill -9 python3.12"
mpirun --allow-run-as-root --hostfile $1 --pernode bash -c  "pkill -9 pt_main_thread"
mpirun --allow-run-as-root --hostfile $1 --pernode bash -c  "pkill -9 python3"
mpirun --allow-run-as-root --hostfile $1 --pernode bash -c  "pkill -9 python"

