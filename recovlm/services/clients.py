#!/usr/bin/env python3
import json
import requests

class PidInfoClient:
    """简化版PID信息服务客户端"""
    
    def __init__(self, host='10.84.241.154', port=8000, timeout=30):
        """初始化客户端
        
        Args:
            host: 服务主机地址，默认为localhost
            port: 服务端口，默认为8000
            timeout: 请求超时时间（秒），默认30秒
        """
        self.base_url = f'http://{host}:{port}/pid_info'
        self.timeout = timeout

    def get_pid_info(self, pid, downloader_params=None, text_params=None):
        """获取指定PID的信息
        
        Args:
            pid: 内容ID
            downloader_params: 下载器的可选参数，如 {'verbose': True}
            text_params: 文本检索的可选参数，如 {'cache_only': True}
            
        Returns:
            包含PID信息的字典
        """
        # 如果有自定义参数，使用POST请求
        if downloader_params is not None or text_params is not None:
            data = {'pid': int(pid)}
            if downloader_params is not None:
                data['downloader_params'] = downloader_params
            if text_params is not None:
                data['text_params'] = text_params
                
            response = requests.post(self.base_url, json=data, timeout=self.timeout)
        else:
            # 否则使用GET请求
            response = requests.get(f'{self.base_url}?pid={pid}', timeout=self.timeout)
        
        # 返回JSON响应
        return response.json()

def demo():
    """演示如何使用PidInfoClient"""
    # 创建客户端实例
    client = PidInfoClient(timeout=10)
    
    # 测试PID
    for test_pid in ["157659349065", "155527081681"]:
    
        # 1. 基本调用（GET方法）
        print("1. 基本调用示例:")
        
        # 不传入任何参数就是默认策略
        # 先取视频，没有视频取10帧
        result = client.get_pid_info(test_pid)

        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 2. 带参数调用（POST方法）
        print("\n2. 带参数调用示例:")
        result = client.get_pid_info(
            test_pid,

            # 1.downloader_params跟离线服务参数一样
            #  want_head: bool = False, # 封面
            #  want_mid: bool = True, # 中间帧
            #  want_tail: bool = False, # 尾帧
            #  mid_frame_num: int = 10, # 中间帧数量
            #  sample_strategy: SampleStrategy = SampleStrategy.EVEN_DIVIDED) -> Optional[List[str]]:

            # 2. need_video = False 表示不需要视频，然后按照want_head, want_mid, want_tail, mid_frame_num, sample_strategy等参数（如果给定了）获取图片
            downloader_params={"mid_frame_num": 20, "need_video": False},
            text_params={}
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    demo()
