import boto3
# import cv2
import traceback
import numpy as np
import torch
import torch.distributed as dist
from botocore import UNSIGNED
from botocore.config import Config
from torchvision.transforms import v2
from torchvision.io import decode_jpeg
from functools import lru_cache 
from tqdm import tqdm
import concurrent
import asyncio


def _remove_slotid_from_pid(pid_sign):
    pid_sign = pid_sign & ((1<<48)-1)
    return pid_sign

debug_dict = {"BlobStoreClient": None}



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
        max_content_length=None,
):
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
        #global debug_dict

        #debug_dict['_get_cover'] = debug_dict.get("_get_cover", 0) + 1
        #if debug_dict['_get_cover'] % 10 == 0: print(debug_dict)
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
            #debug_dict['_get_cover_failed'] = debug_dict.get("_get_cover_failed", 0) + 1
            if self.verbose:
                print("failed get cover", key, e)
            cover = None
        return cover

    def _get_frame(self, key):
        #print("get frame", key)
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
        #global debug_dict
        # debug_dict["all"] = debug_dict.get("all", 0 ) + 1 
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