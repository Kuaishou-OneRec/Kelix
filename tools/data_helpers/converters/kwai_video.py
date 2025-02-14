import os
import json
import argparse
import subprocess
import tempfile
import numpy as np
from typing import Dict, Optional
from .converter import ConverterBase
from recovlm.utils.blobstore_client import BlobStoreClient

class KwaiVideoDownloader(object):

    def __init__(self, video_dir: str, ffmpeg_args: str, caller: str = "recovlm_kwai_video_downloader"):
        self.video_dir = video_dir
        self.ffmpeg_args = list(ffmpeg_args.split(" "))
        self.client = BlobStoreClient(caller=caller)
    
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
    
    def prepare_video(self, photo_id) -> bool:
        video_bytes = self.client.get_video(photo_id)
        if video_bytes is None:
            return None
        output_file = os.path.join(self.video_dir, f"{photo_id}.mp4")
        valid = False
        if not os.path.exists(output_file):
            valid = self.process_video(video_bytes, output_file)
        else:
            valid = True
        if valid:
            return output_file
        else:
            return None

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
                            "text": src['caption']
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

class KwaiWenJuanCaptionConverter(ConverterBase, KwaiVideoDownloader):

    def __init__(self, prompts, source, output_file_path: Optional[str] = None, **kwargs):
        """
        初始化 KwaiWenJuanCaptionConverter 类

        参数:
            prompts: 提示信息列表
            source: 数据来源标识
            output_file_path: 输出 meta 数据文件的路径，如果提供则将 meta 内容追加写入该文件
            kwargs: 传递给父类 KwaiVideoDownloader 的其他参数
        """
        # 调用父类 KwaiVideoDownloader 的初始化方法
        KwaiVideoDownloader.__init__(self, **kwargs)
        self.prompts = prompts
        self.source = source
        # 保存 output_file_path 配置
        self.output_file_path = output_file_path

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        """
        处理输入数据并生成 meta 数据

        参数:
            src: 包含视频、文本等数据的字典，包括 photo_id、caption、ocr、asr、user_comment 和 wenjuan_type

        返回:
            如果视频处理成功则返回包含 meta JSON 数据的字典，否则返回 None
        """
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
                            "text": "视频的标题是：" + src["caption"]
                        },
                        {
                            "type": "text",
                            "text": "视频的ocr 内容是：" + src["ocr"]
                        },
                        {
                            "type": "text",
                            "text": "视频的asr 内容是：" + src["asr"]
                        },
                        {
                            "type": "text",
                            "text": "站内用户的评论内容是：" + " ".join(src["user_comment"])
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
                            "text": ("用户的满意度结果是：满意") if src['wenjuan_type'] == '问卷优质' else ("用户的满意度结果是：不满意")
                        }
                    ]
                }
            ]
            meta = {
                "source": self.source,
                "messages": messages,
            }
            print("meta", meta)

            # 如果设置了 output_file_path，则将 meta 数据追加写入文件，避免MPI模式下覆盖数据
            if self.output_file_path:
                try:
                    with open(self.output_file_path, "a", encoding="utf-8") as f:
                        # 追加 JSON 数据并换行，方便后续处理
                        f.write(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
                except Exception as e:
                    print(f"写入文件 {self.output_file_path} 时出错: {e}")

            return {
                "json": json.dumps(meta)
            }
        else:
            return None
