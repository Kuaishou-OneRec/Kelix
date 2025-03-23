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
import uuid
import time
import io
import base64
import random
import glob
from PIL import Image
import argparse
import json
from collections import Counter
from math import gcd
from copy import deepcopy
from datetime import datetime, timedelta
from functools import partial
sys.setrecursionlimit(2000)


prompts = [
    "请根据视频内容生成一系列用户评论，每条评论都是对上一条评论的回复，形成一条线性的对话链。",
    "假设你是观众，观看视频后发表评论，并与其他观众互动。生成若干条评论，每条评论都是对前一条的回复。",
    "根据视频的抽帧信息，模拟用户之间的对话，生成一系列有逻辑关联的评论，每条评论都是对上一条的回应。",
    "请生成一段用户评论链，每条评论都与视频内容相关，并且每条评论都是对前一条评论的回复，形成连贯的对话。",
    "根据视频内容，生成一系列用户评论，模拟真实用户之间的互动，确保每条评论都是对上一条的回复，形成线性对话。",
    "假设你是一名观众，观看视频后发表评论，并与其他观众进行讨论。生成若干条评论，每条评论都是对前一条的回复，保持对话的连贯性。",
    "根据视频的抽帧信息，生成一系列用户评论，模拟用户之间的互动，确保每条评论都是对上一条的回复，形成线性对话链。",
    "请根据视频内容生成一段用户评论链，每条评论都与视频内容相关，并且每条评论都是对前一条评论的回复，形成连贯的对话。",
    "假设你是观众，观看视频后发表评论，并与其他观众互动。生成若干条评论，每条评论都是对前一条的回复，保持对话的连贯性和逻辑性。",
    "根据视频内容，生成一系列用户评论，模拟真实用户之间的互动，确保每条评论都是对上一条的回复，形成线性对话链，且评论内容与视频内容紧密相关。",
    "看完这个视频，你觉得大家会怎么评论？写几条评论，每条都接着上一条说，像聊天一样。",
    "假设你在刷视频，看到这个内容，你会怎么评论？再想想别人会怎么回复你，写几条连续的评论。",
    "根据视频内容，随便写几条评论，每条都接着上一条说，就像大家在评论区聊起来了一样。",
    "你觉得观众看完这个视频会聊啥？写几条评论，每条都回复上一条，像真实评论区那样。",
    "看完这个视频，你会怎么评论？再想想别人会怎么回你，写几条连续的评论，像聊天一样。",
    "根据视频内容，写几条评论，每条都接着上一条说，就像大家在评论区互动一样。",
    "假设你在评论区，看到这个视频，你会说啥？别人又会怎么回你？写几条连续的评论。",
    "看完这个视频，你觉得评论区会聊啥？写几条评论，每条都接着上一条说，像真实互动一样。",
    "根据视频内容，随便写几条评论，每条都回复上一条，就像大家在评论区聊起来了一样。",
    "你觉得观众看完这个视频会怎么评论？写几条连续的评论，每条都接着上一条说，像聊天一样。",
]


def add_one_day(date_str):
    date_obj = datetime.strptime(date_str, '%Y%m%d')

    new_date_obj = date_obj + timedelta(days=1)
    new_date_str = new_date_obj.strftime('%Y%m%d')
    return new_date_str


def resize_image(image_path, max_size=480):
    img = Image.open(image_path)
    
    width, height = img.size
    
    if width > height:
        ratio = max_size / width
    else:
        ratio = max_size / height
    
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img


def b64image(path):
    image = resize_image(path)
    byte_arr = io.BytesIO()
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image.save(byte_arr, format='JPEG')
    byte_arr = byte_arr.getvalue()
    base64_str = base64.b64encode(byte_arr).decode('utf-8')
    return base64_str


def format_video(video_dir, pid):
    return osp.join(video_dir, "{}.mp4".format(pid))


def exists(path):
    return osp.exists(path) and osp.isfile(path) and osp.getsize(path) > 0


def sub_at(text):
    pattern = r"@.{1,12}\(O[\w-]{8,20}\)"
    result = re.sub(pattern, "", text)
    return result.strip()


def replace_repeated_brackets(text):
    pattern = r'(\[[^\[\]]{1,4}\])\1{2,}'

    result = re.sub(pattern, r'\1\1', text)
    return result


def replace_repeated(text):
    pattern = r'(.)\1{3,}'

    result = re.sub(pattern, r'\1\1\1', text)
    return result


def most_common(text):
    counter = Counter(text)

    most_common = counter.most_common(1)
    return most_common[0] if most_common else None


def is_valid_cmt(cmt):
    cmt = sub_at(cmt).strip()

    cmt = replace_repeated_brackets(cmt)

    if len(cmt) == 0:
        return False

    if len(cmt) < 5:
        return False

    mcm = most_common(cmt)
    if mcm is not None:
        mcm = mcm[0]
        if len(cmt) > 0 and cmt.count(mcm) / len(cmt) > 0.9:
            return False

    return True


def replace_repeated_brackets_empty(text):
    pattern = r'(\[[^\[\]]{1,4}\])\1{1,}'

    result = re.sub(pattern, r'', text)
    return result.strip()


def rewrite_content(content):
    content = sub_at(content)
    content = replace_repeated_brackets(content)
    content = replace_repeated(content)

    content = content.replace("\n", " ")

    while content.count("  ") > 0:
        content = content.replace("  ", " ")

    return content


class Sample(object):

    def __init__(self, photo, user, cmt_id, cmt, reply_to, root, like, reply, delete, mnless):
        self.photo = photo
        self.user = user
        self.cmt_id = cmt_id
        self.cmt = cmt
        self.reply_to = reply_to
        self.root = root
        self.like = like
        self.reply = reply
        self.delete = delete
        self.mnless = mnless
        
    def copy(self):
        return deepcopy(self)
    
    def setattrs(self, **kwargs):
        for name in kwargs:
            value = kwargs[name]
            setattr(self, name, value)
        return self

    def update(self, args_dict):
        return self.setattrs(**args_dict)


class Node(object):
    
    def __init__(self, node_id, sample):
        self.like_tot = 0
        self.reply_tot = 0
        self.size = 0
        self.layer = -1
        self.depth = 1
        self.sample = sample     
        self.children = list()
        self.fat = -1
        self.node_id = node_id

    def add_child(self, node_id, node):
        self.children.append(node_id)
        node.fat = self.node_id

    def copy(self):
        return deepcopy(self)
    
    def router(self, nodes, mode):
        if self.children is None or len(self.children) == 0:
            return None
        assert mode in ["sample", "longest"]
        if mode == "sample":
            s = sum([nodes[child].like_tot for child in self.children])
            if s != 0:
                p = [nodes[child].like_tot / s for child in self.children]
                idx = np.random.choice(list(range(len(self.children))), p=p)
            else:
                idx = np.random.choice(len(self.children))
            return self.children[idx]
        idx, _ = max(enumerate([nodes[child].depth for child in self.children]), key=lambda x: x[1])
        return self.children[idx]


class Tree(object):

    def __init__(self):
        self.root = None
        self.now_node_id = 0
        self.nodes = list()
        self.cmt2node = dict()

    def assign_node(self, sample):
        node = Node(self.now_node_id, sample)
        self.nodes.append(node)
        self.cmt2node[sample.cmt_id] = self.now_node_id
        self.now_node_id += 1
        return node
    
    def exist_or_assign(self, sample):
        cmt_id = sample.cmt_id
        if cmt_id in self.cmt2node:
            return self.nodes[self.cmt2node[cmt_id]]
        return self.assign_node(sample)

    def update_like_reply_tot(self):
        
        assert self.root is not None
        assert self.root.node_id == 0
        visit = [False] * len(self.nodes)

        def dfs(now, layer):
            node = self.nodes[now]

            like_tot = 0
            reply_tot = 0
            size = 0
            depth = 1
            for child in node.children:
                dfs(child, layer + 1)
                like_tot += self.nodes[child].like_tot
                reply_tot += self.nodes[child].reply_tot
                size += self.nodes[child].size
                depth = max(depth, self.nodes[child].depth + 1)
            node.like_tot = like_tot + node.sample.like
            node.reply_tot = reply_tot + node.sample.reply
            node.size = size + 1
            node.layer = layer
            node.depth = depth
            visit[now] = True

        for node_id in range(len(self.nodes)):
            if visit[node_id] is False:
                dfs(node_id, 0)

    def from_pairs(self, root: Sample, pairs):
        rnode = self.exist_or_assign(root)
        for fat, child in pairs:
            self.add_pair(fat, child)

        self.root = rnode
        return self

    def add_pair(self, fat, child):
        fnode = self.exist_or_assign(fat)
        cnode = self.exist_or_assign(child)
        fnode.add_child(cnode.node_id, cnode)

        return self

    def connect_sub_forests(self):
        assert self.root is not None
        assert self.root.node_id == 0

        for node in self.nodes:
            if node.node_id != 0 and node.fat == -1:
                fat = self.root.sample.copy()
                child = node.sample.copy()
                self.add_pair(fat, child)
        
        return self

    def build_text_from_sample(self, mode):

        self.update_like_reply_tot()
        nodes = deepcopy(self.nodes)
        # assert nodes[0]
        que = [0]

        sample_list = list()
        while que:
            top = que.pop(0)
            node = nodes[top]
            sample = node.sample.copy().update(
                {
                    "like_tot": node.like_tot,
                    "reply_tot": node.reply_tot,
                    "is_leaf": len(node.children) == 0,
                    "is_root": node.node_id == 0,
                    "size": node.size,
                    "depth": node.depth
                }
            )
            sample_list.append(sample)
            child = node.router(nodes, mode)
            if child is not None:
                que.append(child)

        return sample_list

    def build_text_from_like_tot(self):

        self.update_like_reply_tot()
        nodes = deepcopy(self.nodes)
        nodes.sort(key=lambda node: -node.like_tot)
        
        sample_list = list()
        for node in nodes:
            sample = node.sample.copy()
            sample = sample.update(
                {
                    "like_tot": node.like_tot,
                    "reply_tot": node.reply_tot,
                    "is_leaf": len(node.children) == 0,
                    "is_root": node.node_id == 0,
                    "size": node.size,
                    "depth": node.depth
                }
            )
            sample_list.append()
        return sample_list

    def build_text_from_layer(self):

        sample_list = list()
        self.update_like_reply_tot()

        root_list = [node.node_id for node in self.nodes if node.fat == -1]
        root_list.sort()
        
        for root in root_list:
            que = [(root, 0)]
            layer = -1
            while que:
                if que[0][1] != layer:
                    que.sort(key=lambda item: (-self.nodes[item[0]].like_tot, -self.nodes[item[0]].reply_tot))

                top, layer = que.pop(0)
                node = self.nodes[top]
                node.layer = layer
                sample = node.sample.copy()
                sample = sample.update(
                    {
                        "like_tot": node.like_tot,
                        "reply_tot": node.reply_tot,
                        "is_leaf": len(node.children) == 0,
                        "is_root": node.node_id == 0,
                        "size": node.size,
                        "depth": node.depth
                    }
                )
                sample_list.append(sample)
                for child in self.nodes[top].children:
                    que.append((child, layer + 1))
        return sample_list
    
    def check_is_not_forest(self):
        if len(self.nodes) > 0:
            root_list = [node.node_id for node in self.nodes if node.fat == -1]
            if len(root_list) == 1 and root_list[0] == 0:
                return True
        return False


def read_rows_in_file(args, fn, is_cache=False):
    parquet_file = pq.ParquetFile(fn)
    if is_cache:
        df = parquet_file.read().to_pandas()
        return df
    df = parquet_file.read(columns=["photo_id", "comment_user_id", "quality_comment_type", "comment_id", "comment_content", "reply_to_comment_id", "root_comment_id", "comment_like_cnt", "comment_reply_cnt", "is_delete", "is_meaningless_comment"]
    ).to_pandas()

    df["main_comment_id"] = df.apply(
        lambda row: row["root_comment_id"] if row["root_comment_id"] != 0 else row["comment_id"],
        axis=1
    )

    df.rename(columns={
        "photo_id": "photo",
        "comment_user_id": "user",
        "comment_id": "id",
        "comment_content": "content",
        "reply_to_comment_id": "reply_to",
        "main_comment_id": "root",
        "comment_like_cnt": "like",
        "comment_reply_cnt": "reply",
        "quality_comment_type": "quality",
        "is_delete": "delete",
        "is_meaningless_comment": "mnless",
    }, inplace=True)

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


def read_hdfs_folder(args, data_folder, postfix, is_cache=False):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if rank == 0:
        print("[INFO] Loading {}...".format(data_folder))
        fn_list = shell_hdfs_ls(data_folder)
        all_files = [fn for fn in fn_list if fn.endswith(postfix)]
        
        shard_size = world_size // gcd(len(all_files), world_size)

        files = [
            (file, shard_id, shard_size)
            for shard_id in range(shard_size)
            for file in all_files
        ]

        # if args.debug:
        #     files = files[:5]
    else:
        files = None

    files = comm.bcast(files, root=0)

    files = files[rank::world_size]

    print("RANK[{}] has {} files to load.".format(rank, len(files)))

    df_list = list()
    for file_path, shard_id, shard_size in tqdm(files):
        df = read_rows_in_file(args, file_path, is_cache=is_cache)
        df_list.append(df.iloc[shard_id::shard_size])
    if len(df_list) == 0:
        return None
    df = pd.concat(df_list, axis=0)
    df.reset_index(drop=True, inplace=True)
    return df


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
    # exit()


def save(args, df, output_dir):

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


def alltoall_by_split(args, df, split_size):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    df_list = list()

    iterator = tqdm(range(split_size), postfix="In alltoall...") if rank == 0 else range(split_size)

    for split_id in iterator:
        pdf = df.iloc[split_id::split_size].copy()
        send_dfs = [pdf[pdf["photo"] % world_size == r] for r in range(world_size)]

        recv_dfs = comm.alltoall(send_dfs)

        pdf = pd.concat(recv_dfs, axis=0, ignore_index=True)
        df_list.append(pdf)
    df = pd.concat(df_list,  axis=0, ignore_index=True)
    return df


def read_and_alltoall(args, folder, name, postfix, is_cache=False):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    df = read_hdfs_folder(args, folder, postfix, is_cache=is_cache)
    
    print(f"Rank[{rank}] has {len(df)} samples before alltoall.")
    df = alltoall_by_split(args, df, 20)
    # send_dfs = [df[df["photo"] % world_size == r] for r in range(world_size)]

    # recv_dfs = comm.alltoall(send_dfs)

    # df = pd.concat(recv_dfs, axis=0, ignore_index=True)

    print(f"Rank[{rank}] has {len(df)} samples after alltoall.")
    if rank == 0:
        print("[INFO]", list(df.columns))

    if not is_cache:
        quailty_df = df[df["quality"].str.lower() != "unknown"].copy()

        qdf = quailty_df[["id"]].copy().rename(columns={"id": "root"})

        branch_df = pd.merge(df, qdf, how="inner", on="root")

        df = pd.concat([quailty_df, branch_df], axis=0, ignore_index=True)

        print(f"Rank[{rank}] has {len(df)} samples after refine.")

    return df


def shuffle_cross_rank(args, df):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    df["seed"] = list(np.random.choice(range(world_size), size=len(df), replace=True))
    send_dfs = [df[df["seed"] == r] for r in range(world_size)]

    recv_dfs = comm.alltoall(send_dfs)

    df = pd.concat(recv_dfs, axis=0)

    df = df.drop('seed', axis=1)
    return df


def parse_images(args, pid):
    pid_str = str(pid)
    files = glob.glob(osp.join(args.image_dir, pid_str[-4:], pid_str, "*.jpg"))
    if len(files) == 0:
        return None, None
    files.sort()
    bases = list()
    images = dict()
    for file in files:
        base = osp.basename(file)
        try:
            b64 = b64image(file)
        except:
            b64 = None
        if b64 is not None:
            bases.append(base)
            images[base] = b64
    
    if len(bases) == 0:
        return None, None
    return bases, images


def build_video_messages(comments, path):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "你是一位短视频内容理解专家。\n"
                },
                {
                    "type": "video",
                    "video": path
                },
                {
                    "type": "text",
                    "text": np.random.choice(prompts)
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": comments
                }
            ]
        }
    ]


def sample_format(args, comments, pid, source):
    video_dir = args.video_dir
    video_path = format_video(video_dir, pid)
    if exists(video_path):
        messages = build_video_messages(comments, video_path)
        sample = {
            "source": source,
            "images": json.dumps(dict()),
            "videos": json.dumps([video_path]),
            "messages": json.dumps(messages),
            "segments": json.dumps(None),
            "metadata": json.dumps({"pid": str(pid)}),
            "uuid": str(uuid.uuid1())
        }
        return sample
    bases, images = parse_images(args, pid)
    if bases is not None:
        messages = build_video_messages(comments, bases)
        sample = {
            "source": source,
            "images": json.dumps(images),
            "videos": json.dumps(None),
            "messages": json.dumps(messages),
            "segments": json.dumps(None),
            "metadata": json.dumps({"pid": str(pid)}),
            "uuid": str(uuid.uuid1())
        }
        return sample
    return None


# def read_cache(args, cache_dir):
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     world_size = comm.Get_size()

#     folder = osp.join(cache_dir, "rank-{}".format(rank))

#     files = shell_hdfs_ls(folder)
#     files = [fn for fn in files if fn.endswith("parquet")]

#     df_list = list()
#     for file in tqdm(files, postfix="In rank {}, read cache...".format(rank)):
#         parquet_file = pq.ParquetFile(file)
#         df = parquet_file.read().to_pandas()
#         df_list.append(df)
    
#     df = pd.concat(df_list, axis=0, ignore_index=False)
#     return df

    
def main(args):
    source = "Comment-Session-Pretrain"
    folder = args.folder
    mode = args.mode

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    now_date = args.start_date
    if not args.load_cache:
        if now_date != "":
            df_list = list()
            while now_date <= args.end_date:
                data_folder = osp.join(folder, "p_date={}".format(now_date))
                df = read_and_alltoall(args, data_folder, "comment", args.postfix)
                df["p_date"] = int(now_date)
                if rank == 0:
                    print("[INFO]", list(df.columns))
                df_list.append(df)
                now_date = add_one_day(now_date)
            df = pd.concat(df_list, axis=0, ignore_index=True)

            df = df.iloc[df.groupby("id")["p_date"].idxmax()]
            df = df.reset_index(drop=True)
            df = df.drop("p_date", axis=1)
        else:
            df = read_and_alltoall(args, folder, "comment", args.postfix)
    else:
        assert args.cache_dir != ""
        df = read_and_alltoall(args, args.cache_dir, "comment", "parquet", is_cache=True)

    # if args.cache_dir != "":
    #     save(args, df, args.cache_dir)
    print(f"Rank[{rank}] has {len(df)} samples building samples.")
    samples = list()

    for _, row in tqdm(df.iterrows(), postfix=f"In rank {rank}, building samples..."):
        sample = Sample(row["photo"], row["user"], row["id"], row["content"], row["reply_to"], row["root"], row["like"], row["reply"], row["delete"], row["mnless"])
        samples.append(sample)

    samples_dict = {
        sample.cmt_id: sample
        for sample in samples
    }

    print(f"Rank[{rank}] has {len(samples)} samples building groups.")
    groups = dict()
    for sample in tqdm(samples, postfix=f"In rank {rank}, building groups..."):
        root = sample.root
        if root not in groups:
            groups[root] = dict()
            groups[root]["root"] = samples_dict[root].copy()
            groups[root]["pairs"] = list()

        if sample.reply_to != 0 and sample.reply_to in samples_dict:
            fat = samples_dict[sample.reply_to].copy()
            child = sample.copy()
            groups[root]["pairs"].append((fat, child))

    print(f"Rank[{rank}] has {len(groups)} samples building trees.")
    trees = dict()

    invalid_cnt = 0
    total_cnt = 0
    for key in tqdm(groups, postfix=f"In rank {rank}, building trees..."):
        tree = Tree().from_pairs(**groups[key])
        if tree.check_is_not_forest() is True:
            trees[key] = tree
        else:
            invalid_cnt += 1
        total_cnt += 1
    
    # if not args.debug:
    assert invalid_cnt <= 10

    print(f"Rank[{rank}] has {len(trees)} samples building comments.")
    samples = list()
    for key in tqdm(trees, postfix=f"In rank {rank}, building comment..."):
        sample_list = trees[key].build_text_from_sample(mode)
        # print(len)
        cmt_list = list()
        cmt_length = -1
        for sample in sample_list:
            content = sample.cmt
            mnless = sample.mnless
            if mnless:
                break
            if not is_valid_cmt(content):
                break
            content = rewrite_content(content)
            if replace_repeated_brackets_empty(content) == "":
                break
            cmt_list.append(content)
            cmt_length += 1 + len(content)
            if cmt_length >= args.max_text_len:
                break
        if len(cmt_list) <= 1:
            continue
        samples.append(
            {
                "id": key,
                "comments": "\n".join(cmt_list)
            }
        )

    print(f"Rank {rank} has {len(samples)} samples after comments.")
    if rank == 0:
        print("[INFO]", list(df.columns))
    
    pdf = pd.DataFrame(samples)
    df = pd.merge(df[["id", "photo"]].copy(), pdf, on="id", how="inner")
    print(f"Rank {rank} has {len(df)} samples after merge.")

    # pids_set = list(set(df.photo.values))

    samples = list()

    for _, row in tqdm(df.iterrows(), postfix="In rank {}...".format(rank)):
        pid = row.photo
        comments = row.comments
        sample = sample_format(args, comments, pid, source)
        # except:
        #     sample = None
        
        if sample is not None:
            samples.append(sample)
        
        if args.debug:
            if len(samples) >= 100:
                break

    df = pd.DataFrame(samples)
    # df = df[["source", "images", "videos", "messages", "segments", "metadata", "uuid"]]

    if args.shuffle:
        print("RANK[{}] has {} rows before all to all.".format(rank, len(df)))
        df = shuffle_cross_rank(args, df)
        print("RANK[{}] has {} rows after all to all.".format(rank, len(df)))

    print(f"[Final]There are {len(df)} samples in rank {rank}")
    save(args, df, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help="hdfs folder")
    parser.add_argument('--output', type=str, help="output path")
    parser.add_argument('--start_date', type=str, default="", help="start date")
    parser.add_argument('--end_date', type=str, default="", help="end date")
    parser.add_argument('--mode', type=str, default="longest", choices=["sample", "longest"], help="end date")
    parser.add_argument('--split_size', type=int, default=536870912, help="split size")
    parser.add_argument('--postfix', type=str, default=".parquet", help="file postfix")
    parser.add_argument('--max_text_len', type=int, default=100000000, help="max text length")
    parser.add_argument('--shuffle', action="store_true", help="shuffle before save")
    parser.add_argument('--image_dir', default="/llm_reco/luoxinchen/dataset/InHouse/Image/pretrain", help="image dir")
    parser.add_argument('--video_dir', default="/llm_reco/luoxinchen/dataset/InHouse/Photo/20250215/480p_60s_4fps_v2", help="video dir")
    parser.add_argument('--cache_dir', default="", help="cache dir")
    parser.add_argument('--debug', action="store_true", help="is debug")
    parser.add_argument('--load_cache', action="store_true", help="is load cache") 
    args = parser.parse_args()
    if args.cache_dir != "":
        assert args.load_cache is True
    main(args)