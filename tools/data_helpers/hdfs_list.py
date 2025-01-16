import argparse
import pyarrow as pa

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=str)
args = parser.parse_args()

fs = pa.hdfs.connect(user='mpi')
files = fs.ls(args.dir)

for fn in files:
    print(fn)