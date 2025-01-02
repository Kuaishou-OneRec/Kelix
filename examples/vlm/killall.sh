mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pkill -9 python3"
mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pkill -9 pt_main_thread" 
mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pkill -9 pt_data_worker" 