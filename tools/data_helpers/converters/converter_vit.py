import argparse
import os
import os.path as osp
import glob
import uuid
import json
import subprocess
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
from mpi4py import MPI
from tqdm import tqdm


def read_rows_in_file(fn):
    parquet_file = pq.ParquetFile(fn)
    df = parquet_file.read().to_pandas()

    return df


def shell_hdfs_ls(source_dir):
    try:
        command = f"hdfs dfs -ls {source_dir}"
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        files = list()
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) > 0 and parts[-1].startswith('viewfs://'):
                files.append(parts[-1])
        return files

    except subprocess.CalledProcessError as e:
        return list()


def read_hdfs_folder(data_folder, postfix, output):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if rank == 0:
        fn_list = shell_hdfs_ls(data_folder)
        files = [fn for fn in fn_list if fn.endswith(postfix)]
    else:
        files = None
    comm.Barrier()
    files = comm.bcast(files, root=0)
    files = files[rank::world_size]

    print("Has {} files to load.".format(len(files)))

    if len(files) == 0:
        return None
    for file_path in tqdm(files):
        df = read_rows_in_file(file_path)
        samples = list()
        for _, row in tqdm(df.iterrows(), total=len(df), postfix="In rank {}".format(rank)):
            keys = list(json.loads(row["images"]).keys())[0]
            image = json.loads(row["images"])[keys]
            sample = {
                "source": row["source"],
                "task": "caption",
                "images": json.dumps([image]),
                "videos": json.dumps(list()),
                "text": json.loads(row["messages"])[-1]["content"][0]["text"],
                #"text": json.loads(row["segments"])[1]["text"],
                "metadata": json.dumps(None),
                "uuid": str(uuid.uuid1()),
            }
            samples.append(sample)
        processed_df = pd.DataFrame(samples)
        save(processed_df, output)
    return None


def build_empty_df(df):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if rank == 0:
        assert df is not None
        empty = df.iloc[:1].copy()
        empty = empty.drop(empty.index)
    else:
        empty = None

    empty = comm.bcast(empty, root=0)
    if df is not None:
        return df
    return empty


def flush(fs, buffer, temp_dir, output_dir):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    tempfile = os.path.join(temp_dir, f"rank-{rank}-{str(uuid.uuid1())}.parquet")
    filename = os.path.join(output_dir, f"rank-{rank}-{str(uuid.uuid1())}.parquet")

    df = pd.DataFrame(buffer)
    pq.write_table(pa.Table.from_pandas(df, nthreads=1), tempfile)

    if fs is not None:
        fs.mv(tempfile, filename)
    else:
        command = "mv {} {}".format(tempfile, filename)
        subprocess.run(command, shell=True, check=True)

    print(f"RANK[{rank}] write to {filename} success")


def save(df, output):

    output_dir = output
    temp_dir = osp.join(output_dir, "tmp")

    rank = 0

    if output_dir.startswith("viewfs://"):
        fs = pa.hdfs.connect(user="mpi")
        if rank == 0:
            fs.mkdir(output_dir)
            fs.mkdir(temp_dir)
    else:
        os.makedirs(temp_dir, exist_ok=True)
        fs = None

    buffer = list()

    for _, row in df.iterrows():
        sample = row.to_dict()
        buffer.append(sample)
        if len(buffer) >= 512:
            flush(fs, buffer, temp_dir, output_dir)
            buffer = list()

    if len(buffer) > 0:
        flush(fs, buffer, temp_dir, output_dir)


def main(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    read_hdfs_folder(args.folder, args.postfix, args.output)

    # samples = list()
    # for _, row in tqdm(df.iterrows(), total=len(df), postfix="In rank {}".format(rank)):
    #     keys = list(json.loads(row["images"]).keys())[0]
    #     image = json.loads(row["images"])[keys]
    #     sample = {
    #         "source": row["source"],
    #         "task": "caption",
    #         "images": json.dumps([image]),
    #         "videos": json.dumps(list()),
    #         "text": json.loads(row["messages"])[-1]["content"][0]["text"],
    #         #"text": json.loads(row["segments"])[1]["text"],
    #         "metadata": json.dumps(None),
    #         "uuid": str(uuid.uuid1()),
    #     }
    #     samples.append(sample)
    # df = pd.DataFrame(samples)
    # save(df, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--postfix', type=str, default="parquet")
    parser.add_argument('--output', type=str, required=True)
    ags = parser.parse_args()
    main(ags)
