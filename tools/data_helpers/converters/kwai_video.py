import os
import os.path as osp
import json
import argparse
import subprocess
import tempfile
import numpy as np
import random
from typing import Dict, List, Tuple, Set, Optional, List
from .converter import ConverterBase
from recovlm.utils.blobstore_client import BlobStoreClient
import glob
import uuid
import cv2
import base64
from PIL import Image
import traceback
import re
import random

class KwaiVideoDownloader(object):

    def __init__(self, ffmpeg_args: str,
        video_dir: str = "/llm_reco/luoxinchen/dataset/InHouse/Photo/20250215/480p_60s_4fps_v2",  
        image_dir: str = '/llm_reco/luoxinchen/dataset/InHouse/Image/pretrain',
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
    def _encode_image(self, image_path: str) -> str:
        """
        将图片编码为base64字符串
        
        Args:
            image_path: 图片路径
            
        Returns:
            base64编码的图片字符串
        """
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image: {image_path}")
                return None

            # 调整图片大小
            image_resized = cv2.resize(image, (224, 224))

            # 编码为JPEG
            _, encoded_image = cv2.imencode(".jpg", image_resized)

            # 转换为base64
            return base64.b64encode(encoded_image).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def prepare_video(self, photo_id) -> Optional[str]:
        self.data["total"] += 1
        output_file = os.path.join(self.video_dir, f"{photo_id}.mp4")
        
        res_video = None

        # Check if file already exists and is valid
        if os.path.exists(output_file):
            print(f"find {output_file}, abort")
            res_video = output_file
            return res_video
        video_bytes = None
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

        return None

    def prepare_image(self, photo_id):
        images = list()
        checkfile = os.path.join(self.image_dir, str(photo_id)[-4:], f"{photo_id}")
        # output_file = os.path.join(self.image_dir, f"{photo_id}.jpg")

        # Check if file already exists and is valid
        if os.path.exists(checkfile):
            print(f"find {checkfile}, abort")
            images = glob.glob(os.path.join(checkfile, '*.jpg'))
            images.sort()
            print(f"Found {len(images)} .jpg files in {checkfile}")
            # try:
            #     images = [os.path.abspath(file_path) for file_path in jpg_files]
            # except Exception as e:
            #     print(f"Error retrieving image for {photo_id}: {e}")
            #     images = []

        else:
            print(f"Directory {checkfile} does not exist.")
            return None

            # res_image = output_file
            # return res_image
        videolist=[]
        imagedic = {}
        for image in images:
            imagebyte = self._encode_image(image)
            if imagebyte == None:
                continue 
            temp = {
                                "type": "image",
                                "image": image
                            }
            videolist.append(temp)
            imagedic[image]=imagebyte

        if videolist==[]:
            return None,None
        return videolist,imagedic



# class KwaiVideoDownloader(object):

#     def __init__(self, video_dir: str, ffmpeg_args: str, caller: str = "recovlm_kwai_video_downloader"):
#         self.video_dir = video_dir
#         self.ffmpeg_args = list(ffmpeg_args.split(" "))
#         self.client = BlobStoreClient(caller=caller)
    
#     def process_video(self, input_bytes, output_file):
#         with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp_input_file:
#             temp_input_file.write(input_bytes)
#             temp_input_file_path = temp_input_file.name
#             process = subprocess.Popen(
#                 [
#                     'ffmpeg',
#                     '-i', temp_input_file_path,
#                     *self.ffmpeg_args,
#                     output_file
#                 ],
#                 stdin=subprocess.PIPE,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE
#             )
#             stdout, stderr = process.communicate()

#             if process.returncode != 0:
#                 # print(f"ffmpeg error: {stderr.decode('utf-8')}")
#                 return False
#             else:
#                 return True
    
#     def prepare_video(self, photo_id) -> bool:
#         video_bytes = self.client.get_video(photo_id)
#         if video_bytes is None:
#             return None
#         output_file = os.path.join(self.video_dir, f"{photo_id}.mp4")
#         valid = False
#         if (not os.path.exists(output_file)) or (osp.getsize(output_file) == 0):
#             valid = self.process_video(video_bytes, output_file)
#         else:
#             valid = True
#         if valid:
#             return output_file
#         else:
#             return None


class KwaiVideoShuffleConverter(ConverterBase, KwaiVideoDownloader):

    def __init__(
        self,
        prompts,
        source,
        **kwargs
    ):
        KwaiVideoDownloader.__init__(self, **kwargs)
        self.prompts = prompts
        self.source = source
        self.image_dir = '/llm_reco/zangdunju/dataset/reorder/frames'


def shuffle_with_indices(self, sorted_list):
    # 创建带有原索引的列表
    indexed_list = list(enumerate(sorted_list))
    # 随机打乱该列表
    random.shuffle(indexed_list)
    # 提取打乱后的元素列表
    shuffled_list = [item[1] for item in indexed_list]
    # 创建字典记录每个原索引对应的打乱后的位置（从1开始）
    position_map = {original_idx: shuffled_pos + 1 for shuffled_pos, (original_idx, _) in enumerate(indexed_list)}
    # 生成原列表各元素在打乱后的位置列表
    original_indices = [position_map[i] for i in range(len(sorted_list))]
    return shuffled_list, original_indices
    

    def fetch_image(self,photo_id):
        checkfile = os.path.join(self.image_dir, str(photo_id)[-4:], f"{photo_id}")
        if not os.path.exists(checkfile):
            return {},[]
        images = glob.glob(os.path.join(checkfile, '*.jpg'))
        images.sort()
        if len(images)==0:
            return {},[]
        imagedic = {}
        
        for image in images:
            imagebyte = self._encode_image(image)
            if imagebyte == None:
                continue 
            imagedic[image]=imagebyte
        return imagedic,images


    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        photo_id = src['photo_id']
        photo_id = str(photo_id)
        
        imagedic,images = self.fetch_image(photo_id)
        if len(images)==0:
            return None
        simages,ranklist = self.shuffle_with_indices(images)
        prompt = np.random.choice(self.prompts)
        content = []
        ranklist = [str(i) for i in ranklist]
        text = ','.join(ranklist)
        for image in simages:
            content.append({
                "type":"image",
                "image":image
            })
        content.append({
            "type":"text",
            "text":prompt

        })

        if len(images)>0:
            
            messages = [
                {
                    "role": "user",
                    "content": content
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
            print(messages)
            meta = {
                "source": self.source,
                "images": json.dumps(imagedic),
                "videos": json.dumps([]),
                "segments": None,
                "metadata": None,
                "messages": json.dumps(messages),
                "uuid": str(uuid.uuid1())
            }
            
            return meta
        else:
            return None








class KwaiVideoTitleCaptionConverter(ConverterBase, KwaiVideoDownloader):

    def __init__(
        self,
        prompts,
        source,
        **kwargs
    ):
        KwaiVideoDownloader.__init__(self, **kwargs)
        self.prompts = prompts
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        photo_id = src['photo_id']
        photo_id = str(photo_id)
        if src['title'] is None and src['caption_clean'] is None:
            return None
        title = src['title'] if src['title'] is not None else ''
        caption_clean = src['caption_clean'] if src['caption_clean'] is not None else ''
        text = title + caption_clean
        filename = self.prepare_video(photo_id)
        ##=====video
        if filename is not None:
            prompt = np.random.choice(self.prompts)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": filename
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
            meta = {
                "source": self.source,
                "images": json.dumps({}),
                "videos": json.dumps([filename]),
                "segments": None,
                "metadata": None,
                "messages": json.dumps(messages),
                "uuid": str(uuid.uuid1())
            }
            #print("meta", meta)
            return meta
        ##======video

        ##======image
        else:
            filename,images = self.prepare_image(photo_id)
            if filename is not None and len(filename)!=0:
                print('found in image~~~!!!!')
                prompt = np.random.choice(self.prompts)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": filename
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": text
                            }
                        ]
                    }
                ]
                #print("meta", meta)
                meta = {
                "source": self.source,
                "images": json.dumps(images),
                "videos": json.dumps([]),
                "segments": None,
                "metadata": None,
                "messages": json.dumps(messages),
                "uuid": str(uuid.uuid1())
                }
                return meta
            else:
                return None
            ##======image



class KwaiVideoClickAfterShowConverter(ConverterBase, KwaiVideoDownloader):

    def __init__(
        self,
        prompts,
        source,
        **kwargs
    ):
        KwaiVideoDownloader.__init__(self, **kwargs)
        self.prompts = prompts
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        photo_id = src['photo_id']
        photo_id = str(photo_id)
        if src['keyword'] is None:
            return None
        filename = self.prepare_video(photo_id)
        ##=====video
        if filename is not None:
            prompt = np.random.choice(self.prompts)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": filename
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": src['keyword']
                        }
                    ]
                }
            ]
            meta = {
                "source": self.source,
                "images": json.dumps({}),
                "videos": json.dumps([filename]),
                "segments": None,
                "metadata": None,
                "messages": json.dumps(messages),
                "uuid": str(uuid.uuid1())
            }
            #print("meta", meta)
            return meta
            
        ##======video

        ##======image
        else:
            filename,images = self.prepare_image(photo_id)
            if filename is not None and len(filename)!=0:
                print('found in image~~~!!!!')
                prompt = np.random.choice(self.prompts)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": filename
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": src['keyword']
                            }
                        ]
                    }
                ]
                meta = {
                "source": self.source,
                "images": json.dumps(images),
                "videos": json.dumps([]),
                "segments": None,
                "metadata": None,
                "messages": json.dumps(messages),
                "uuid": str(uuid.uuid1())
                }
                #print("meta", meta)
                return meta
            else:
                return None
            ##======image


class KwaiVideoClickAfterShow10Converter(ConverterBase, KwaiVideoDownloader):

    def __init__(
        self,
        prompts,
        source,
        **kwargs
    ):
        KwaiVideoDownloader.__init__(self, **kwargs)
        self.prompts = prompts
        self.source = source
    def clean(self,keylist):
        seen = set()
        cleankeylist = []
        for key in keylist:
            if key not in seen:
                seen.add(key)
                cleankeylist.append(key)
        return cleankeylist
            
    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        photo_id = src['photo_id']
        photo_id = str(photo_id)
        if src['keyword_1'] is None:
            return None
        keylist = []
        for i in range(1,11):
            if src[f'keyword_{i}'] is None:
                break 
            keylist.append(src[f'keyword_{i}'])
        if keylist == []:
            return None 
        #print(keylist)
        keylist = self.clean(keylist)
        #print('clean',keylist)
        text = ','.join(keylist)

        filename = self.prepare_video(photo_id)
        ##=====video
        if filename is not None:
            prompt = np.random.choice(self.prompts)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": filename
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
            meta = {
                "source": self.source,
                "images": json.dumps({}),
                "videos": json.dumps([filename]),
                "segments": None,
                "metadata": None,
                "messages": json.dumps(messages),
                "uuid": str(uuid.uuid1())
            }
            #print("meta", meta)
            return meta
            
        ##======video

        ##======image
        else:
            filename,images = self.prepare_image(photo_id)
            if filename is not None and len(filename)!=0:
                print('found in image~~~!!!!')
                prompt = np.random.choice(self.prompts)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": filename
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": text
                            }
                        ]
                    }
                ]
                meta = {
                "source": self.source,
                "images": json.dumps(images),
                "videos": json.dumps([]),
                "segments": None,
                "metadata": None,
                "messages": json.dumps(messages),
                "uuid": str(uuid.uuid1())
                }
                #print("meta", meta)
                return meta
            else:
                return None
            ##======image




class KwaiVideoCategoryConverter(ConverterBase, KwaiVideoDownloader):

    def __init__(
        self,
        prompts,
        source,
        **kwargs
    ):
        KwaiVideoDownloader.__init__(self, **kwargs)
        self.prompts = prompts
        self.source = source
        os.system("pip install selectolax")
    def catgen(self,firstn,firstp,secondn,secondp,thridn,thridp,fourthn,fourthp):
        text = ''
        prob = 1
        if firstn=='UNKNOWN':
            return text
        prob*=firstp
        if prob<0.5:
            return text
        text+=firstn
        text+=','

        if secondn=='UNKNOWN':
            return text[:-1]
        prob*=secondp
        if prob<0.5:
            return text[:-1]
        text+=secondn
        text+=','

        if thridn=='UNKNOWN':
            return text[:-1]
        prob*=thridp
        if prob<0.5:
            return text[:-1]
        text+=thridn
        text+=','

        if fourthn=='UNKNOWN':
            return text[:-1]
        prob*=fourthp
        if prob<0.5:
            return text[:-1]
        text+=fourthn
        return text

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        photo_id = src['photo_id']
        photo_id = str(photo_id)
        first_level_category_name = src['first_level_category_name']
        first_level_category_prob = src['first_level_category_prob']
        second_level_category_name = src['second_level_category_name']
        second_level_category_prob = src['second_level_category_prob']
        third_level_category_name = src['third_level_category_name']
        third_level_category_prob = src['third_level_category_prob']
        fourth_level_category_name = src['fourth_level_category_name']
        fourth_level_category_prob = src['fourth_level_category_prob']
        text = self.catgen(first_level_category_name,
        first_level_category_prob,
        second_level_category_name,
        second_level_category_prob,
        third_level_category_name,
        third_level_category_prob,
        fourth_level_category_name,
        fourth_level_category_prob)
        if text == '':
            return
        filename = self.prepare_video(photo_id)
        ##=====video
        if filename is not None:
            prompt = np.random.choice(self.prompts)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": filename
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
            meta = {
                "source": self.source,
                "images": json.dumps({}),
                "videos": json.dumps([filename]),
                "segments": None,
                "metadata": None,
                "messages": json.dumps(messages),
                "uuid": str(uuid.uuid1())
            }
            #print("meta", meta)
            return meta
        ##======video

        ##======image
        else:
            filename,images = self.prepare_image(photo_id)
            if filename is not None and len(filename)!=0:
                print('found in image~~~!!!!')
                prompt = np.random.choice(self.prompts)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": filename
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": text
                            }
                        ]
                    }
                ]
                #print("meta", meta)
                meta = {
                "source": self.source,
                "images": json.dumps(images),
                "videos": json.dumps([]),
                "segments": None,
                "metadata": None,
                "messages": json.dumps(messages),
                "uuid": str(uuid.uuid1())
                }
                return meta
            else:
                return None
            ##======image



class KwaiVideoCaptionConverter(ConverterBase, KwaiVideoDownloader):
    def __init__(
        self,
        prompts,
        source,
        **kwargs
    ):
        KwaiVideoDownloader.__init__(self, **kwargs)
        self.prompts = prompts
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        photo_id = src['photo_id']
        filename = self.prepare_video(photo_id)
        if filename is not None:
            prompt = np.random.choice(self.prompts)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": filename
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ['caption']
                        }
                    ]
                }
            ]
            meta = {
                "source": self.source,
                "messages": messages,
            }
            print("meta", meta)
            return {
                "json": json.dumps(meta)
            }
        else:
            return None

class KwaiWenJuanCaptionVideoConverter(ConverterBase, KwaiVideoDownloader):

    def __init__(self, prompts, source: Optional[str] = None, **kwargs):
        """
        初始化 KwaiWenJuanCaptionVideoConverter 类

        参数:
            prompts: 提示信息列表
            source: 数据来源标识
            kwargs: 传递给父类 KwaiVideoDownloader 的其他参数
        """
        # 调用父类 KwaiVideoDownloader 的初始化方法
        KwaiVideoDownloader.__init__(self, **kwargs)
        self.prompts = prompts
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        """
        处理输入数据并生成 meta 数据

        参数:
            src: 包含视频、文本等数据的字典，包括 photo_id、caption、ocr、asr、user_comment 和 wenjuan_type

        返回:
            如果视频处理成功则返回包含 meta JSON 数据的字典，否则返回 None
        """
        try:
            # 获取基础字段，使用空字符串替代None值
            photo_id = str(src.get('photo_id', ''))  # 确保photo_id是字符串
            caption = src.get('caption', '')
            ocr = src.get('ocr', '')
            asr = src.get('asr', '')
            user_comment = src.get('user_comment', [])
            
            # 确保user_comment是列表类型
            if isinstance(user_comment, str):
                user_comment = [user_comment]
            elif user_comment is None:
                user_comment = []

            filename = self.prepare_video(photo_id)
            if filename is not None:
                prompt = np.random.choice(self.prompts)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": filename
                            },
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "text",
                                "text": "视频的标题是：" + str(caption)
                            },
                            {
                                "type": "text",
                                "text": "视频的ocr 内容是：" + str(ocr)
                            },
                            {
                                "type": "text",
                                "text": "视频的asr 内容是：" + str(asr)
                            },
                            {
                                "type": "text",
                                "text": "站内用户的评论内容是：" + " ".join(str(c) for c in user_comment)
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": ("用户的满意度结果是：满意") if src.get('wenjuan_type') == '问卷优质' else ("用户的满意度结果是：不满意")
                            }
                        ]
                    }
                ]
                meta = {
                    "source": self.source,
                    "messages": messages,
                }
                return {
                    "json": json.dumps(meta, ensure_ascii=False)  # 确保UTF-8编码
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error in converter for photo_id {src.get('photo_id', 'unknown')}: {str(e)}")
            print(traceback.format_exc())
            return None



class KwaiWenJuanCaptionFrameConverter(ConverterBase, KwaiVideoDownloader):

    def __init__(self, prompts, source, frame_dir: Optional[str] = None, 
                 cot_txt_file_path: Optional[str] = None, test_id_file_path: Optional[str] = None,
                 max_frames_per_video: int = None, enable_debug: bool = False, 
                 enable_cmt_to_cot: bool = False, enable_llm_response: bool = False, **kwargs):
        """
        初始化 KwaiWenJuanCaptionFrameConverter 类

        参数:
            prompts: 提示信息列表
            source: 数据来源标识
            frame_dir: 视频抽帧结果存储目录
            cot_txt_file_path: 包含LLM响应的txt文件目录
            test_id_file_path: 测试集photo_id文件路径
            max_frames_per_video: 每个视频最多获取的帧数，None表示获取全部
            enable_debug: 是否启用调试日志
            enable_llm_response: 是否启用LLM响应
            enable_cmt_to_cot: 是否将站内用户评论转换为cot格式
            kwargs: 传递给父类的其他参数
        """
        video_dir = kwargs.pop('video_dir')
        ffmpeg_args = kwargs.pop('ffmpeg_args')
        KwaiVideoDownloader.__init__(self, video_dir, ffmpeg_args)
        self.prompts = prompts
        self.source = source
        self.frame_dir = frame_dir
        self.cot_txt_file_path = cot_txt_file_path
        self.max_frames_per_video = max_frames_per_video
        self.enable_debug = enable_debug
        self.enable_cmt_to_cot = enable_cmt_to_cot
        # 缓存LLM响应
        self.enable_llm_response = enable_llm_response
        if self.enable_llm_response:
            self.llm_responses = self._load_llm_responses() if cot_txt_file_path else {}
        else:
            self.llm_responses = {}
        # 加载测试集ID
        self.test_ids = self._load_test_ids(test_id_file_path) if test_id_file_path else set()
        # 建立图片索引
        self.frame_index = self._build_frame_index() if frame_dir else {}
        
        if self.enable_debug:
            print(f"Initialized with {len(self.frame_index)} frame entries")




        
    def _build_frame_index(self) -> Dict[str, List[str]]:
        """
        建立图片索引，将所有jpg文件按photo_id分组
        
        Returns:
            Dict[str, List[str]]: photo_id到图片路径列表的映射
        """
        frame_index = {}
        if not self.frame_dir:
            return frame_index

        try:
            # 遍历frame_dir下所有子目录
            for root, _, files in os.walk(self.frame_dir):
                for file in files:
                    if file.endswith('.jpg'):
                        try:
                            # 从文件名提取photo_id，保持为字符串类型
                            photo_id = file.split('_')[0]
                            full_path = os.path.join(root, file)
                            
                            # 将路径添加到对应photo_id的列表中
                            if photo_id not in frame_index:
                                frame_index[photo_id] = []
                            frame_index[photo_id].append(full_path)
                        except Exception as e:
                            if self.enable_debug:
                                print(f"Error processing file {file}: {e}")
                            continue

            # 对每个photo_id的图片列表进行排序
            for photo_id in frame_index:
                frame_index[photo_id].sort(key=self._safe_frame_number)

            if self.enable_debug:
                print(f"Built frame index with {len(frame_index)} photo_ids")
                
        except Exception as e:
            print(f"Error building frame index: {e}")
            
        return frame_index

    def _safe_frame_number(self, filename: str) -> int:
        """
        安全地从文件名中提取帧号
        """
        try:
            frame_str = os.path.basename(filename).split('_')[-1].split('.')[0]
            return int(frame_str)
        except (ValueError, IndexError):
            if frame_str == 'single':
                return 0
            return float('inf')

    def _get_frame_images(self, photo_id: str) -> List[str]:
        """
        从索引中获取指定photo_id的抽帧图片路径
        
        Args:
            photo_id: 视频ID（可能是int或str类型）
            
        Returns:
            图片路径列表,按帧顺序排序
        """
        # 确保photo_id是字符串类型
        photo_id_str = str(photo_id)
        
        # 直接从索引中获取图片路径
        frame_files = self.frame_index.get(photo_id_str, [])
        
        if self.enable_debug and not frame_files:
            print(f"No frames found for photo_id {photo_id} (str: {photo_id_str})")
            print(f"Available photo_ids in index: {list(self.frame_index.keys())[:5]}...")
        
        # 如果设置了最大帧数限制，则只返回指定数量的帧
        if self.max_frames_per_video is not None and len(frame_files) > self.max_frames_per_video:
            frame_files = frame_files[:self.max_frames_per_video]
        
        return frame_files

    def _load_test_ids(self, file_path: str) -> set:
        """
        从文件加载测试集photo_id
        
        Args:
            file_path: 测试集ID文件路径
            
        Returns:
            set: 测试集photo_id集合
        """
        test_ids = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 跳过第一行（字段名）
                next(f)
                # 读取所有ID
                for line in f:
                    photo_id = line.strip()
                    if photo_id:
                        test_ids.add(photo_id)
            if self.enable_debug:
                print(f"====KwaiWenJuanCaptionFrameConverter====\nLoaded {len(test_ids)} test IDs from {file_path}")
        except Exception as e:
            print(f"Error loading test IDs from {file_path}: {e}")
        return test_ids

    def _extract_satisfaction_label(self, content: str) -> Optional[str]:
        """
        从LLM响应中提取满意度标签
        """
        # 如果内容为空，返回None
        if not content:
            return None
        
        # 定义可能的满意度标签模式
        satisfaction_patterns = [
            r"【结果[:：]满意】",
            r"【结果[:：]不满意】",
            r"【結果：满意】",
            r"【結果：不满意】",
            r"【满意】",
            r"【不满意】",
            r"\*\*结果\*\*[:：]满意",
            r"\*\*结果\*\*[:：]不满意",
            r"结果[:：]\s*满意",
            r"结果[:：]\s*不满意"
        ]
        
        # 遍历所有可能的模式
        for pattern in satisfaction_patterns:
            match = re.search(pattern, content)
            if match:
                return "不满意" if "不满意" in match.group() else "满意"
            
        # 如果仍然没有找到，返回None
        return None

    def _load_llm_responses(self) -> Dict[str, str]:
        """
        从txt文件加载LLM响应
        """
        responses = {}
        if not os.path.exists(self.cot_txt_file_path):
            return responses

        # 修改正则表达式以同时匹配单引号和双引号的情况
        content_pattern = re.compile(r"['\"]content['\"]: ['\"](.+?)['\"](,\s*['\"]tool_calls['\"]|,\s*['\"]logprobs['\"])")
        total_line = 0

        for txt_file in glob.glob(os.path.join(self.cot_txt_file_path, "*.txt")):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        total_line += 1
                        if "photo_id is" in line and "response is" in line:
                            try:
                                # 提取 photo_id
                                photo_id = line.split("photo_id is")[1].split(",")[0].strip()
                                
                                # 尝试解析完整的响应 JSON
                                response_str = line.split("response is")[1].strip()
                                try:
                                    response_data = json.loads(response_str.replace("'", '"'))
                                    if 'choices' in response_data and len(response_data['choices']) > 0:
                                        content = response_data['choices'][0]['message']['content']
                                        # 提取满意度标签
                                        label = self._extract_satisfaction_label(content)
                                        if label:
                                            responses[photo_id] = {
                                                'content': content,
                                                'label': label
                                            }
                                        else:
                                            print(f"json 解析方式 没有找到photo_id: {photo_id} 的LLM label, response is {content}")
                                except json.JSONDecodeError:
                                    # 如果JSON解析失败，尝试使用正则表达式
                                    match = content_pattern.search(line)
                                    if match:
                                        content = match.group(1)
                                        label = self._extract_satisfaction_label(content)
                                        if label:
                                            responses[photo_id] = {
                                                'content': content,
                                                'label': label
                                            }
                                        else:
                                            print(f"没有找到photo_id: {photo_id} 的LLM label, response is {content}")
                                    else:
                                        print(f"正则表达式 没有找到photo_id: {photo_id} 的LLM响应, response is {line}")
                            except Exception as e:
                                print(f"处理行时出错: {e}")
                                continue
            except Exception as e:
                print(f"处理文件 {txt_file} 时出错: {e}")
                continue
        
        if self.enable_debug:
            print(f"====KwaiWenJuanCaptionFrameConverter====\nLoad LLM cot succ, total line {total_line}, total result {len(responses)}")

        return responses

    def _encode_image(self, image_path: str) -> str:
        """
        将图片编码为base64字符串
        
        Args:
            image_path: 图片路径
            
        Returns:
            base64编码的图片字符串
        """
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image: {image_path}")
                return None
                
            # 调整图片大小
            image_resized = cv2.resize(image, (224, 224))
            
            # 编码为JPEG
            _, encoded_image = cv2.imencode(".jpg", image_resized)
            
            # 转换为base64
            return base64.b64encode(encoded_image).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        """
        处理输入数据并生成数据项
        
        Args:
            src: 包含视频信息的字典,包括photo_id、caption、ocr、asr、user_comment等字段
            
        Returns:
            处理后的数据项,包含images、messages等信息，如果是测试集ID则返回None
        """
        try:
            photo_id = src['photo_id']
            caption = src.get('caption', '')
            ocr = src.get('ocr', '')
            asr = src.get('asr', '')
            user_comment = src.get('user_comment', [])
            is_correct_response = True
            
            # 如果启用了标签过滤
            if self.enable_llm_response:
                wenjuan_label = "满意" if src['wenjuan_type'] == '问卷优质' else "不满意"
                llm_response = self.llm_responses.get(str(photo_id))
                
                # 如果找到了LLM响应但标签不匹配，跳过该样本
                if llm_response and llm_response['label'] != wenjuan_label:
                    is_correct_response = False
                    

            # # 如果是测试集ID，直接返回None
            # if str(photo_id) in self.test_ids:
            #     if self.enable_debug:
            #         print(f"Skipping test ID: {photo_id}")
            #     return None
                
            # Use the video file returned by prepare_video
            filename = self.prepare_video(photo_id)
            if filename is None:
                if self.enable_debug:
                    print(f"No video file found for photo_id: {photo_id}")
                return None

            # Construct the content list with the video file

            content_list = [
                {
                    "type": "video",
                    "video": filename
                },
                {"type": "text", "text": self.prompts[0]},
                {"type": "text", "text": f"视频的标题是：{caption}"},
                {"type": "text", "text": f"视频的ocr 内容是：{ocr}"},
                {"type": "text", "text": f"视频的asr 内容是：{asr}"}
            ]
            # if not self.enable_cmt_to_cot:
            #     content_list.append({"type": "text", "text": f"站内用户的评论内容是：{'<comment>'.join(user_comment)}"})

            messages = [
                {
                    "role": "user",
                    "content": content_list
                }
            ]

            assistnat_content = []
            # if self.enable_cmt_to_cot:
            #     assistnat_content.append({"type": "text", "text": f"站内用户的评论内容是：{'<comment>'.join(user_comment)}"})
            # 添加 cot 结果
            if self.enable_llm_response:
                if str(photo_id) in self.llm_responses and is_correct_response:
                    assistnat_content.append({
                        "type": "text",
                        "text": str(self.llm_responses[str(photo_id)]['content'])
                    })
                else:
                    # 如果打开了 cot 并且没有找到正确结果，则返回 None
                    return None
            else:
                assistnat_content.append({
                    "type": "text",
                    "text": "【结果：满意】" if src['wenjuan_type'] == '问卷优质' else "【结果：不满意】"
                })

            messages.append({
                "role": "assistant",
                "content": assistnat_content
            })

            # 构建返回数据，确保所有JSON序列化使用UTF-8
            meta = {
                "images": json.dumps(None, ensure_ascii=False),
                'videos': json.dumps([filename], ensure_ascii=False),
                "messages": json.dumps(messages, ensure_ascii=False),
                'segments': json.dumps(None, ensure_ascii=False),
                "source": str(self.source),
                "metadata": None,
                "uuid": str(uuid.uuid1())
            }

            return meta

        except Exception as e:
            print(f"Error processing photo_id {src.get('photo_id', 'unknown')}: {str(e)}")
            print(traceback.format_exc())
            return None

class i2iConverter(ConverterBase, KwaiVideoDownloader):
    def __init__(self, prompts, source, 
                 cot_txt_file_path: Optional[str] = None, 
                 max_frames_per_video: int = None, enable_debug: bool = False, 
                 enable_cmt_to_cot: bool = False, enable_llm_response: bool = False, **kwargs):
        """
        初始化 KwaiWenJuanCaptionFrameConverter 类

        参数:
            prompts: 提示信息列表
            source: 数据来源标识
            frame_dir: 视频抽帧结果存储目录
            cot_txt_file_path: 包含LLM响应的txt文件目录
            test_id_file_path: 测试集photo_id文件路径
            max_frames_per_video: 每个视频最多获取的帧数，None表示获取全部
            enable_debug: 是否启用调试日志
            enable_llm_response: 是否启用LLM响应
            enable_cmt_to_cot: 是否将站内用户评论转换为cot格式
            kwargs: 传递给父类的其他参数
        """
        video_dir = kwargs.pop('video_dir')
        ffmpeg_args = kwargs.pop('ffmpeg_args')
        KwaiVideoDownloader.__init__(self, video_dir, ffmpeg_args)
        self.prompts = prompts
        self.source = source
        self.cot_txt_file_path = cot_txt_file_path
        self.max_frames_per_video = max_frames_per_video
        self.enable_debug = enable_debug
        self.enable_cmt_to_cot = enable_cmt_to_cot
        # 缓存LLM响应

        




    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        """
        Process input data and generate meta data

        Parameters:
            src: Dictionary containing video and text data including photoId, caption, title, text, ocr, and asr

        Returns:
            A dictionary containing meta JSON data if video processing is successful, otherwise None
        """
        try:
            enable_reverse = random.random() < 0.5
            src_pid = str(src.get('src_pid',''))
            src_caption = str(src.get('src_caption',''))
            src_title = str(src.get('src_title',''))
            src_text = str(src.get('src_text',''))
            src_ocr = str(src.get('src_ocr',''))
            src_asr = str(src.get('src_asr',''))
            sim_pid = str(src.get('sim_pid',''))
            sim_caption = str(src.get('sim_caption',''))
            sim_title = str(src.get('sim_title',''))
            sim_text = str(src.get('sim_text',''))
            sim_ocr = str(src.get('sim_ocr',''))
            sim_asr = str(src.get('sim_asr',''))
            neg_pid = str(src.get('neg_pid',''))
            neg_caption = str(src.get('neg_caption',''))
            neg_title = str(src.get('neg_title',''))
            neg_text = str(src.get('neg_text',''))
            neg_ocr = str(src.get('neg_ocr',''))
            neg_asr = str(src.get('neg_asr',''))
            if enable_reverse:
                sim_pid, neg_pid = neg_pid, sim_pid
                sim_caption, neg_caption = neg_caption, sim_caption
                sim_title, neg_title = neg_title, sim_title
                sim_text, neg_text = neg_text, sim_text
                sim_ocr, neg_ocr = neg_ocr, sim_ocr
                sim_asr, neg_asr = neg_asr, sim_asr

            src_video_filename = self.prepare_video(src_pid)
            sim_video_filename = self.prepare_video(sim_pid)
            neg_video_filename = self.prepare_video(neg_pid)
            if src_video_filename is None or sim_video_filename is None or neg_video_filename is None:
                return None
            content_list = [
                {
                    "type": "text", 
                    "text": self.prompts[0]
                },
                {
                    "type": "text",
                    "text": "源视频的图像内容是："
                },
                {
                    "type": "video", 
                    "video": src_video_filename
                },
                {
                    "type": "text",
                    "text": "源视频的OCR内容是：" + src_ocr
                },
                {
                    "type": "text",
                    "text": "源视频的ASR内容是：" + src_asr
                },
                {
                    "type": "text",
                    "text": "源视频的标题是：" + src_title
                },
                {
                    "type": "text",
                    "text": "视频1的图像内容是："
                },
                {
                    "type": "video", 
                    "video": sim_video_filename
                },
                {
                    "type": "text",
                    "text": "视频1的OCR内容是：" + sim_ocr
                },
                {
                    "type": "text",
                    "text": "视频1的ASR内容是：" + sim_asr
                },
                {
                    "type": "text",
                    "text": "视频1的标题是：" + sim_title
                },
                {
                    "type": "text",
                    "text": "视频2的图像内容是："
                },
                {
                    "type": "video", 
                    "video": neg_video_filename
                },
                {
                    "type": "text",
                    "text": "视频2的OCR内容是：" + neg_ocr
                },
                {
                    "type": "text",
                    "text": "视频2的ASR内容是：" + neg_asr
                },
                {
                    "type": "text",
                    "text": "视频2的标题是：" + neg_title
                }
            ]

            messages = [
                {
                    "role": "user",
                    "content": content_list
                }
            ]

            assistnat_content = []
            assistnat_content.append({
                "type": "text",
                "text": "和源视频最相似的视频是【视频2】" if enable_reverse else "和源视频最相似的视频是【视频1】"
            })

            messages.append({
                "role": "assistant",
                "content": assistnat_content
            })

            meta = {
                "images": json.dumps([], ensure_ascii=False),
                'videos': json.dumps([], ensure_ascii=False),
                "messages": json.dumps(messages, ensure_ascii=False),
                'segments': json.dumps(None, ensure_ascii=False),
                "source": str(self.source),
                "metadata": None,
                "uuid": str(uuid.uuid1())
            }
            print("debug_log::: meta is ", meta)
            return meta
        except Exception as e:
            print(f"Error processing photoId {src.get('photoId', 'unknown')}: {str(e)}")
            print(traceback.format_exc())
            return None
