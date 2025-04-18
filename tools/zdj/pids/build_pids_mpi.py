import sys
import uuid
import os
import re
import os.path as osp
import subprocess
from mpi4py import MPI
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import time
import argparse
from collections import Counter
from math import gcd
from copy import deepcopy
from datetime import datetime, timedelta


def add_one_day(date_str):
    date_obj = datetime.strptime(date_str, '%Y%m%d')

    new_date_obj = date_obj + timedelta(days=1)
    new_date_str = new_date_obj.strftime('%Y%m%d')
    return new_date_str


def read_rows_in_file(args, fn):
    parquet_file = pq.ParquetFile(fn)
    df = parquet_file.read().to_pandas()

    return df


def save_file(df, fn):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, fn)


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


def save(args, df):

    output_dir = args.output
    temp_dir = osp.join(output_dir, "tmp")

    split_size = args.split_size

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if output_dir.startswith("viewfs://"):
        fs = pa.hdfs.connect(user="mpi")
        if rank == 0:
            fs.mkdir(output_dir)
            fs.mkdir(temp_dir)
    else:
        os.makedirs(temp_dir, exist_ok=True)
        fs = None

    buffer = list()
    buffer_size = 0

    for _, row in df.iterrows():
        sample = row.to_dict()
        buffer.append(sample)
        for k, v in sample.items():
            buffer_size += sys.getsizeof(v)
        if buffer_size >= split_size:
            flush(fs, buffer, temp_dir, output_dir)

            buffer = list()
            buffer_size = 0

    if len(buffer) > 0:
        flush(fs, buffer, temp_dir, output_dir)


def read_hdfs_folder(args, data_folder, file_postfix):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if rank == 0:
        fn_list = shell_hdfs_ls(data_folder)
        files = [fn for fn in fn_list if file_postfix in fn]
        
    else:
        files = None

    files = comm.bcast(files, root=0)

    files = files[rank::world_size]

    print("RANK[{}] has {} files to load.".format(rank, len(files)))

    df_list = list()
    for file_path in tqdm(files):
        df = read_rows_in_file(args, file_path)
        df_list.append(df)
    
    if len(df_list) == 0:
        return None

    df = pd.concat(df_list, axis=0)
    df.reset_index(drop=True, inplace=True)

    return df


def parse_photo_id(args, df):
    
    if 1 == 1:
        # df = df[["photo"]].rename(columns={"photo": "photo_id"})
        df = df[["photo_id", "realshow_count"]]
        df["realshow_count"] = df.apply(lambda s: str(s.realshow_count), axis=1)
    elif 1 == 1:
        contents = list()
        for _, row in tqdm(df.iterrows()):
            if "ai小快" in row.comment_content.lower():
                contents.append(row.comment_content)
            if len(contents) > 10000:
                break
        df = pd.DataFrame(
            {
                "photo_id": contents
            }
        )
    elif 1 == 1:
        # df = df[df["id"] == df["root"]][["id"]].copy()
        df = df[df["upload_dt"] >= "2025-03-17"].reset_index(drop=True)
        df = df[["photo_id"]]
    elif 2 == 1:
        pids = list()
        pid2 = list()
        for _, row in df.iterrows():
            msg = json.loads(row.messages)[0]["content"]
            spid = list()
            for content in msg:
                if content["type"] == "image":
                    ppp = int(content["image"].split("_")[0])
                    if ppp not in spid:
                        spid.append(ppp)
            if len(spid) == 1:
                continue
            assert len(spid) == 2, spid
            spid = list(spid)
            pids.append(spid[0])
            pid2.append(spid[1])
        df = pd.DataFrame({"pid1": pids, "pid2": pid2})
    elif 1 == 2:
        pids = list()
        for _, row in df.iterrows():
            msg = json.loads(row.messages)[0]["content"]
            spid = set()
            for content in msg:
                if content["type"] == "image":
                    spid.add(int(content["image"].split("_")[0]))
            if len(spid) != 1:
                print(spid)
                continue
            # assert len(spid) == 1, spid
            pids.append(list(spid)[0])
        df = pd.DataFrame({"photo_id": pids})
    elif "show_cnt" in df.columns:
        pids = list()
        for x in df.photo_id.values:
            pids.extend(list(x))
        df = pd.DataFrame({"photo_id": pids})
    elif "photo_id_first" in df.columns and "photo_id_second" in df.columns:
        df = pd.DataFrame({"photo_id": list(set(df.photo_id_first.values) | set(df.photo_id_second.values))})
    elif "photo_id" in df.columns:
        df = df[["photo_id"]]
    else:
        df["photo_id"] = df.apply(lambda s: int(osp.basename(osp.splitext(json.loads(s.videos)[0])[0])), axis=1)
        df = df[["photo_id"]]
    
    if args.duplicate:
        df = df.drop_duplicates()

    return df


def read_multi_folder(args, folder, postfix):

    start_date = args.start_date
    end_date = args.end_date

    now_date = start_date

    df_list = list()
    if now_date != "":
        while now_date <= end_date:
            data_folder = osp.join(folder, "p_date={}".format(now_date))
            df = read_hdfs_folder(args, data_folder, postfix)
            if df is not None:
                df_list.append(df)
            now_date = add_one_day(now_date)
        
        df = pd.concat(df_list, axis=0)
    else:
        df = read_hdfs_folder(args, folder, postfix)

    df.reset_index(drop=True, inplace=True)
    return df


def main(args):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    df = read_multi_folder(args, args.folder, args.postfix)

    df = parse_photo_id(args, df)

    df_list = comm.gather(df, root=0)

    if rank == 0:
        df = pd.concat(df_list, axis=0)
        if args.duplicate:
            df = df.drop_duplicates()

        df["realshow_count"] = df.apply(lambda s: int(s.realshow_count), axis=1)
        df = df.groupby('photo_id')['realshow_count'].sum().reset_index()
        df = df.sort_values(by='realshow_count', ascending=False)
        df = df.sample(frac=args.ratio)
        df.reset_index(drop=True, inplace=True)
        save(args, df)

    print("RANK[{}] done".format(rank))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help="hdfs folder")
    parser.add_argument('--output', type=str, help="output path")
    parser.add_argument('--split_size', type=int, default=536870912, help="split size")
    parser.add_argument('--postfix', type=str, default=".parquet", help="file postfix")
    parser.add_argument('--start_date', type=str, help="file postfix")
    parser.add_argument('--end_date', type=str, help="file postfix")
    parser.add_argument('--ratio', type=float, help="sample ratio")
    parser.add_argument('--duplicate', action="store_true", help="")
    args = parser.parse_args()

    main(args)