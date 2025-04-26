sed 's/=1/=8/g' /etc/mpi/hostfile  | head -100 > /etc/mpi/hostfile_seq

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile_seq --pernode bash -c "pip install -U torchao"
mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile_seq --pernode bash -c "pip install omegaconf"
mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile_seq --pernode bash -c "pip install transformers==4.51.3"
