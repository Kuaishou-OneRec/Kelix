import os
import json
import uuid
import argparse
import boto3
import traceback
import torch
import concurrent
import asyncio
import subprocess
import tempfile
import numpy as np
import os.path as osp
import torch.distributed as dist

from botocore import UNSIGNED
from botocore.config import Config
from torchvision.transforms import v2
from torchvision.io import decode_jpeg
from functools import lru_cache 
from typing import Dict, Optional
from tqdm import tqdm

def _remove_slotid_from_pid(pid_sign):
    pid_sign = pid_sign & ((1<<48)-1)
    return pid_sign

debug_dict = {"BlobStoreClient": None}
PhotoBase0316 = "/llm_reco/luoxinchen/dataset/InHouse/Photo/20250215/480p_60s_4fps_0215_0316"
ffmpeg_args = '''-vf scale='if(gt(iw,ih),min(480,iw),-2)':'if(gt(ih,iw),min(480,ih),-2)' -r 4 -t 60 -c:a aac -c:v libx264 -b:a 55k -pix_fmt yuv420p -probesize 200M -analyzeduration 200M'''

def get_video_dir_by_pid(pid: str):
    folder = str(int(pid[-4:]))
    return osp.join(PhotoBase0316, folder, "{}.mp4".format(pid))

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_folder", type=str, default=None, help="The directory of the download data.")
    parser.add_argument("--file_path", type=str, default=None, help="Pid txt file path ")
    parser.add_argument("--retry_num", type=int, default=1, help="Number of retries")
    return parser

class BlobStoreClient(object):
    """
    https://docs.corp.kuaishou.com/d/home/fcABIeySrELpWqukNBCl_NYaB
    用法见_test_demo函数
    """
    def __init__(self, 
        endpoint_url="http://bs3-hb1.internal",
        s3_client_config = Config(
                s3={'addressing_style': 'path'},
                region_name='HB1',
                signature_version=UNSIGNED,
                read_timeout=1,
                connect_timeout=1,
                max_pool_connections=10, # 默认是10，如果thread > 10就会达到最大连接数
                retries = {
                    'max_attempts': 1,
                    'mode': 'standard'
                },
        ),
        predefined_frame_ids=[-1], # 定义秒帧, -1是封面,
        predefined_audio_secs=30,
        thread_num=6,
        caller="end2endreco",
        verbose=False,
        max_content_length=None):
        """
        target_shape 默认图片大小(取不到图片返回这个大小)
        predefined_frame_ids 图片第几秒取一帧，-1表示取封面
        """
        self.client = boto3.client("s3", endpoint_url=endpoint_url, config=s3_client_config)

        def _add_service_header(request, **kwargs):
            request.headers.add_header('service', caller) # 一定要添加服务名标识！
        event_system = self.client.meta.events
        event_system.register('before-sign.*.*', _add_service_header)

        self.predefined_frame_ids = predefined_frame_ids
        self.predefined_audio_secs = predefined_audio_secs

        self.predefined_frame_tags = self._frameid2frametag(self.predefined_frame_ids)

        self.thread_num = thread_num
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_num)
        self.verbose = verbose
        self.max_content_length = max_content_length
    
    def _frameid2frametag(self, frame_ids):
        return [
            '' if fid == -1 else f"_{fid * 30}" for fid in frame_ids
        ]
    
    def get_object(self, bucket, key):
        try:
            rsp = self.client.get_object(Bucket=bucket, Key=key)
            data = rsp['Body'].read()
        except Exception as e:
            print(e)
            data = None
        return data

    def _get_cover(self, key):
        try:
            if ".jpg" in key:
                cover = self.client.get_object(Bucket='photo-def', Key=key) # 返回的图片大小并不固定，而且目前服务不支持，比直接从磁盘上读取慢十倍，暂时不知道有没有QPS限制
            else:
                key += ".jpg"
                cover = self.client.get_object(Bucket='photo-def', Key=key)
            cover = cover['Body'].read()
            if self.verbose:
                print("get cover success", key)
        except Exception as e:
            if self.verbose:
                print("failed get cover", key, e)
            cover = None
        return cover

    def _get_frame(self, key):
        try:
            frame_hw3 = self.client.get_object(Bucket='frame-f', Key=key)
            frame_hw3 = frame_hw3['Body'].read()
            if self.verbose:
                print("get frame success", key)
        except Exception as e:
            if self.verbose:
                print("failed _get_frame", key, e)
            frame_hw3 = None
        return frame_hw3

    def _get_one_image(self, key):
        """
        1. key是id，就获取封面
        2. key是id_secid，获取第secid/30秒
        """
        if "_" not in key:
            try:
                frame_hw3 = self._get_cover(key)
            except:
                frame_hw3 = self._get_frame(key + "_0")
        else:
            frame_hw3 = self._get_frame(key)
        return frame_hw3

    def _get_one_audio(self, key):
        """
        1. key是 id_secid, 获取前secid秒的音频
        grep -c "audio_get_fail" tmp.txt -- 1538
        grep -c "audio_get_success" tmp.txt -- 9377
        16.4% miss

        # ph_a_3_141816958618
        """
        audio_bytes = None
        try: 
            try: 
                audio_bytes = self.client.get_object(Bucket='image-rank-audio', Key=key + '.mp3')['Body'].read()
            except:
                pass
            if audio_bytes is None: 
                audio_bytes = self.client.get_object(Bucket='image-rank-audio', Key=f"ph_a_3_{key.split('_')[0]}")['Body'].read()
        except Exception as e: 
            if self.verbose:
                print("failed get audio", key, e)
        return audio_bytes
    
    def get_video(self, pid, maxsize=None):
        video_bytes = None
        try:
            rsp = self.client.get_object(Bucket='video-def', Key=f"{pid}_b.mp4")
            content_length = rsp['ContentLength']
            if self.max_content_length is not None and content_length > self.max_content_length:
                print(f"{pid} video size {content_length} too large, skip")
                return None
            body = rsp['Body']
            body.set_socket_timeout(10)
            if maxsize is not None:
                video_bytes = body.read(maxsize)
            else:
                video_bytes = body.read()
        except:
            print(traceback.format_exc())
        return video_bytes

    def get_audio(self, pid):
        pid = _remove_slotid_from_pid(pid)
        audio = self._get_one_audio(f"{pid}_{self.predefined_audio_secs}")
        return audio

    def get_all_frames(self, pid, frame_ids=None):
        pid = _remove_slotid_from_pid(pid)
        if frame_ids is None: frame_id_tags = self.predefined_frame_tags
        else: frame_id_tags = self._frameid2frametag(frame_ids)
        frame_list = [self._get_one_image(f"{pid}{tag}") for tag in frame_id_tags]
        return frame_list
    
    def async_get_audio(self, pid):
        return self.executor.submit(self.get_audio, pid)
    
    def async_get_all_frames(self, pid, frame_ids=None):
        return self.executor.submit(self.get_all_frames, pid, frame_ids)


    def get_all_frames_list(self, pids, frame_ids=None, decoder=lambda x: x):
        futures = [self.executor.submit(self.get_all_frames, pid, frame_ids) for pid in pids]
        results = [future.result() for future in tqdm(futures)]  # 保证结果顺序与提交顺序一致
        return results

    def close(self):
        self.client.close()

class KwaiVideoDownloader(object):

    def __init__(self, ffmpeg_args: str,
        video_dir: str = "/llm_reco/luoxinchen/dataset/InHouse/Photo/20250215/480p_60s_4fps_v2",  
        image_dir: str = './tmp',
        caller: str = "recovlm_kwai_video_downloader",
        **kargs):
        self.video_dir = video_dir
        self.image_dir = image_dir
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        self.ffmpeg_args = list(ffmpeg_args.split(" "))
        self.client = BlobStoreClient(caller=caller)
        self.data = {"total": 0, "failed": 0}
    
    def process_video(self, input_bytes, output_file):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp_input_file:
            temp_input_file.write(input_bytes)
            temp_input_file_path = temp_input_file.name
            process = subprocess.Popen(
                [
                    'ffmpeg',
                    '-i', temp_input_file_path,
                    *self.ffmpeg_args,
                    output_file
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"ffmpeg error: {stderr.decode('utf-8')}")
                return False
            else:
                return True
    
    def process_image(self, input_bytes, output_file):
        with open(output_file, 'wb') as f:
            # print(type(input_bytes), output_file); exit()
            f.write(input_bytes)
            return True

    def prepare_video(self, photo_id) -> Optional[str]:
        self.data["total"] += 1
        output_file = os.path.join(self.video_dir, f"{photo_id}.mp4")
        
        res_video = None

        # Check if file already exists and is valid
        if os.path.exists(output_file):
            print(f"find {output_file}, abort")
            res_video = output_file
            return res_video

        try:
            video_bytes = self.client.get_video(photo_id)
        except Exception as e:
            print(f"Error retrieving video for {photo_id}: {e}")
            res_video = None

        if video_bytes is None:
            self.data["failed"] += 1
            print(f"No video found for {photo_id}.")
            res_video = None
        
        # Process video if it doesn't exist
        if video_bytes is not None and self.process_video(video_bytes, output_file):
            res_video = output_file
        
        if res_video is not None:
            return res_video

        return self.prepare_image(photo_id)

    def prepare_image(self, photo_id) -> Optional[str]:
        # 下载图片
        res_image = None
        image_bytes = None
        output_file = os.path.join(self.image_dir, f"{photo_id}.jpg")

        # Check if file already exists and is valid
        if os.path.exists(output_file):
            print(f"find {output_file}, abort")
            res_image = output_file
            return res_image

        try:
            image_bytes = self.client._get_one_image(str(photo_id))
        except Exception as e:
            print(f"Error retrieving image for {photo_id}: {e}")
            res_image = None

        if image_bytes is None:
            self.data["failed"] += 1
            print(f"No image found for {photo_id}.")
            res_image = None

        if image_bytes is not None and self.process_image(image_bytes, output_file):
            res_image = output_file

        return res_image

def get_filenames_without_extension(directories):
    if isinstance(directories, str):
        directories = [directories]
    
    result = {}
    
    for directory in directories:
        if not os.path.exists(directory):
            result[directory] = []
        else:
            result[directory] = [os.path.splitext(file)[0] for file in os.listdir(directory)]
    
    return result

def compare_and_print_missing(lines, downloaded_list):
    # Find elements that are in lines but not in downloaded_list
    missing_elements = [item for item in lines if item not in downloaded_list]
    
    # Print the missing elements
    if missing_elements:
        print(f"Found {len(missing_elements)} missing elements:")
    else:
        print("No missing elements found.")
    
    return missing_elements


if __name__ == "__main__":
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    ffmpeg_args = '''-vf scale='if(gt(iw,ih),min(480,iw),-2)':'if(gt(ih,iw),min(480,ih),-2)' -r 4 -t 60 -c:a aac -c:v libx264 -b:a 55k -pix_fmt yuv420p -probesize 200M -analyzeduration 200M'''

    retry_num=args.retry_num
    download_folder=args.download_folder
    os.makedirs(download_folder, exist_ok=True)
    file_path=args.file_path

    video_folder=f'{download_folder}/tmp_video'
    image_folder=f'{download_folder}/tmp_image'

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file]
    print(f"len of pids: {len(lines)}")

    for line in lines:
        KwaiVideoDownloader(ffmpeg_args, video_folder, image_folder, ).prepare_video(line)
    
    downloaded_result = get_filenames_without_extension([video_folder,image_folder])
    downloaded_list = downloaded_result[video_folder] + downloaded_result[image_folder]
    print(f"{len(downloaded_result[video_folder])} videos, {len(downloaded_result[image_folder])} images, total {len(downloaded_list)}, miss {len(lines)-len(downloaded_list)}")

    missing = compare_and_print_missing(lines, downloaded_list)

    # retry
    while retry_num > 0 and missing:
        retry_num -= 1
        print(f"retrying,  and {retry_num} retry opportunities left")
        for miss in missing:
            KwaiVideoDownloader(ffmpeg_args, video_folder, image_folder, ).prepare_video(miss)
        
        downloaded_result = get_filenames_without_extension([video_folder,image_folder])
        downloaded_list = downloaded_result[video_folder] + downloaded_result[image_folder]
        missing = compare_and_print_missing(lines, downloaded_list)

        print(f"{len(downloaded_result[video_folder])} videos, {len(downloaded_result[image_folder])} images, total {len(downloaded_list)}, miss {len(lines)-len(downloaded_list)}")

    # jsonl file for video only
    jsonl_file = f"{download_folder}/{file_path.split('/')[-1].replace('.txt','.jsonl')}"
    with open(jsonl_file, 'w') as out_file:
        for root, _, files in os.walk(video_folder):
            for file in files:
                video_path = os.path.join(root, file)
                
                # 创建字典数据并直接写入文件
                data = {
                    "video_path": video_path,
                    "prompt": "Describe this video.",
                    "pid_file_name": file_path
                }
                
                out_file.write(json.dumps(data) + '\n')
    
    # 计算处理的文件数量
    video_count = sum(len(files) for _, _, files in os.walk(video_folder))
    print(f"Successfully created jsonl file at {jsonl_file} with {video_count} entries.")