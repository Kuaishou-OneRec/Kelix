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
# os.system("pip install pyltp")
# from pyltp import SentenceSplitter


prompts = [
    "这视频太有意思了！你觉得最搞笑/感动的瞬间是啥？多写几条评论说说看。",
    "看完这视频，你第一反应是啥？用几条评论表达你的感受。",
    "如果你是观众，看完这视频你会怎么评论？可以别太正式，随便聊聊。可以多写几条。",
    "这视频里有啥让你印象深刻的画面？写几条评论分享一下。",
    "你觉得这视频想表达啥？用几条评论总结一下，再带点个人感受。",
    "这视频让你想起了啥？写几条评论，说说你的联想。",
    "如果你是视频里的主角，你会怎么回应观众的评论？写几条评论看看。",
    "这视频让你笑了/感动了/思考了？写几条评论，说说为啥。",
    "看完这视频，你觉得最值得讨论的点是啥？写几条评论带个头。",
    "这视频让你有啥冲动？是想分享还是吐槽？写几条评论表达一下。",
    "请根据视频内容，生成几个简洁的用户评论。",
    "结合视频的抽帧信息，撰写几条评论，表达你对视频主题的深刻印象。",
    "基于视频内容，生成几条评论，探讨视频所传递的核心信息或情感。",
    "请根据视频中的关键画面，撰写几条评论，分享你对视频的独特见解。",
    "结合视频的视觉和叙事元素，生成几条评论，表达你对视频的整体感受。",
    "请根据视频内容，撰写几条评论，分析视频中令人印象深刻的细节或场景。",
]

prompts = [
    "请给这个视频生成几条评论内容。",
    "根据视频内容生成几条相关的评论。",
    "根据视频中的视觉元素，为其生成几条评论，来表达你的看法。",
    "请以专业的视角分析视频中的技术细节，并生成几条评论。",
    "观看视频后，写几条评论，表达你对视频内容的感受。",
    "观看视频后，结合画面内容，写几条评论来描述你的感受。",
    "结合视频内容和上下文信息，为视频生成几条评论。",
    "如果你是一个短视频观看者，在看完上述短视频之后你会有怎样的评论内容，请写出几条。",
    "请扮演一位短视频爱好者，为上述短视频生成几条评论内容来表达你观看后的切身体验。",
]


prompts = [
    "观看视频发表一条优质的评论内容。",
    "根据视频内容生成一条特别优质的评论。",
    "请以专业的视角分析视频中的技术细节，并生成一条优质的评论。",
    "观看视频后，写一条神评论，表达你对视频内容的感受。",
    "观看视频后，结合画面内容，写一条优质评论来描述你的感受。",
    "结合视频内容和上下文信息，为视频生成一条神评论。",
    "如果你是一个短视频观看者，在看完上述短视频之后你会有怎样的评论内容，请写出一条优质评论。",
    "请扮演一位短视频爱好者，为上述短视频生成一条优质评论内容来表达你观看后的切身体验。",
]


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


def sub_all_at(text):
    pattern = r'@快手用户\d{8,20}'
    text = re.sub(pattern, " ", text)
    pattern = r'@\S{2,12}\s'
    text = re.sub(pattern, " ", text)
    return text


def rewrite_content(content):
    content = sub_at(content)
    content = sub_all_at(content)
    content = replace_repeated_brackets(content)
    content = replace_repeated(content)

    content = content.replace("\n", " ")
    while content.count("  ") > 0:
        content = content.replace("  ", " ")

    return content


def read_rows_in_file(args, fn):
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


def read_hdfs_folder(args, data_folder, postfix):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if not data_folder.startswith("viewfs"):
        if rank == 0:
            files = glob.glob(osp.join(data_folder, "*{}".format(postfix)))
        else:
            files = None
        
        files = comm.bcast(files, root=0)
        files = files[rank::world_size]

        print("RANK[{}] has {} files to load.".format(rank, len(files)))

        df_list = list()
        for file_path in tqdm(files):
            df = pd.read_parquet(file_path)
            df_list.append(df)
        if len(df_list) == 0:
            return None
        df = pd.concat(df_list, axis=0, ignore_index=True)
        return df

    if rank == 0:
        fn_list = shell_hdfs_ls(data_folder)
        all_files = [fn for fn in fn_list if fn.endswith(postfix)]
        
        shard_size = world_size // gcd(len(all_files), world_size)

        files = [
            (file, shard_id, shard_size)
            for shard_id in range(shard_size)
            for file in all_files
        ]
    else:
        files = None

    files = comm.bcast(files, root=0)

    files = files[rank::world_size]

    print("RANK[{}] has {} files to load.".format(rank, len(files)))

    df_list = list()
    for file_path, shard_id, shard_size in tqdm(files):
        df = read_rows_in_file(args, file_path)
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


def read_and_alltoall(args, folder, name, postfix, key="photo_id"):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    df = read_hdfs_folder(args, folder, postfix)
    if world_size > 1:
        df = build_empty_df(df)
        
        print(f"Rank[{rank}] has {len(df)} samples before alltoall.")
        send_dfs = [df[df[key] % world_size == r] for r in range(world_size)]

        recv_dfs = comm.alltoall(send_dfs)

        df = pd.concat(recv_dfs, axis=0, ignore_index=True)

        print(f"Rank[{rank}] has {len(df)} samples after alltoall.")
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
    df = df.sample(frac=1.0).reset_index(drop=True)
    return df


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
    prompt = np.random.choice(prompts)
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
                    "text": "\n" + prompt
                }
            ]
        },
        # {
        #     "role": "assistant",
        #     "content": [
        #         {
        #             "type": "text",
        #             "text": comments
        #         }
        #     ]
        # }
    ]


def sample_format(args, comments, pid, source, show_cnt=None):
    video_dir = args.video_dir
    video_path = format_video(video_dir, pid)
    if "480p_60s_4fps_v2" in video_path:
        post = str(int(str(pid)[-4:]))
        video_path = video_path.replace("480p_60s_4fps_v2", "480p_60s_4fps_0215_0316/{}".format(post))
        
    if exists(video_path):
        messages = build_video_messages(comments, video_path)
        sample = {
            "source": source,
            "images": json.dumps(dict()),
            "videos": json.dumps([video_path]),
            "messages": json.dumps(messages),
            "segments": json.dumps(None),
            "metadata": json.dumps({"pid": str(pid), "show_cnt": str(show_cnt)}),
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
            "metadata": json.dumps({"pid": str(pid), "show_cnt": str(show_cnt)}),
            "uuid": str(uuid.uuid1())
        }
        return sample
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


def main(args):
    source = "God-Comment-Reward"
    folder = args.folder

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    df = read_and_alltoall(args, folder, "comment", args.postfix, key="photo")
    # df = build_empty_df(df)
    df = df[df["split"] == args.mode]
    df = df[df["picture"] == 0].reset_index(drop=True)
    
    samples = list()
    for _, row in tqdm(df.iterrows(), postfix=f"In rank {rank}..."):
        pid = row.photo
        if "chosen" in row.to_dict():
            sample = sample_format(args, None, pid, source)
            sample["chosen"] = row.chosen
            sample["rejected"] = row.rejected
            samples.append(sample)
        else:
            negatives = json.loads(row.negative_list)
            if negatives is None:
                continue
            for negative in negatives:
                show_cnt = row.show_cnt
                sample = sample_format(args, None, pid, source, show_cnt=show_cnt)
                chosen = json.dumps(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": row.god_comment
                            }
                        ]
                    }
                )
                rejected = json.dumps(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": negative
                            }
                        ]
                    }
                )
                sample["chosen"] = chosen
                sample["rejected"] = rejected
                samples.append(sample)

    df = pd.DataFrame(samples)
    # df = df[["source", "images", "videos", "messages", "segments", "metadata", "uuid"]]

    if args.shuffle:
        print("RANK[{}] has {} rows before all to all.".format(rank, len(df)))
        df = shuffle_cross_rank(args, df)
        print("RANK[{}] has {} rows after all to all.".format(rank, len(df)))

    print(f"[Final]There are {len(df)} samples in rank {rank}")
    save(args, df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help="hdfs folder")
    parser.add_argument('--output', type=str, help="output path")
    parser.add_argument('--split_size', type=int, default=536870912, help="split size")
    parser.add_argument('--postfix', type=str, default=".parquet", help="file postfix")
    parser.add_argument('--max_text_len', type=int, default=100000000, help="max text length")
    parser.add_argument('--shuffle', action="store_true", help="shuffle before save")
    parser.add_argument('--image_dir', default="/llm_reco/luoxinchen/dataset/InHouse/Image/pretrain", help="image dir")
    parser.add_argument('--video_dir', default="/llm_reco/luoxinchen/dataset/InHouse/Photo/20250215/480p_60s_4fps_v2", help="video dir")
    parser.add_argument('--debug', action="store_true", help="is debug")
    parser.add_argument('--mode', default="train", help="mode")
    args = parser.parse_args()
    main(args)
