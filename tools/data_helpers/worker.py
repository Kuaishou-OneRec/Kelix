import os
from mpi4py import MPI
import webdataset as wds
from utils import MPIBase

class MPITarWriterWorker(MPIBase):

    def __init__(self, config):
        super().__init__()
        self.output_dir = config.output_dir
        self.shard_size = config.shard_size
        if self.rank == 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        self.comm.barrier()

        self._write_shard_id = self.rank
        self._cur_size = 0
        self._tarwriter = None
        self._cur_fn = None
        self._tarwriter_config = config.get("tarwriter", {})
        self._create_new_writer()
    
    def _create_new_writer(self):
        if self._tarwriter is not None:
            self.mpi_print(f"{self._cur_fn} is written")
            self._tarwriter.close()
            self._cur_size = 0
            self._write_shard_id += self.world_size
        self._cur_fn = os.path.join(self.output_dir, f"{self._write_shard_id:05d}.tar")
        self._tarwriter = wds.TarWriter(self._cur_fn, **self._tarwriter_config)
    
    def write_sample(self, sample):
        if "__key__" not in sample:
            sample["__key__"] = f"{self._write_shard_id:06d}{self._cur_size:04d}"
        self._tarwriter.write(sample)
        self._cur_size += 1
        if self._cur_size >= self.shard_size:
            self._create_new_writer()
