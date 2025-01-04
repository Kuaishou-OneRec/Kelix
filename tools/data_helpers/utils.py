from mpi4py import MPI

class MPIBase(object):

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()
    
    def mpi_print(self, *args, **kwargs):
        print(f"RANK[{self.rank}]", *args, **kwargs)