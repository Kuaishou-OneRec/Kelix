from google.protobuf import text_format

from kess.framework import (
    ClientOption,
    GrpcClient,
    KessOption,
)

from tools.gpt_tools.mmu.mmu_chat_gpt_pb2 import MmuChatGptRequest, MmuChatGptResponse
from tools.gpt_tools.mmu.mmu_chat_gpt_pb2_grpc import MmuChatGptServiceStub
from tools.gpt_tools.mmu.media_common_pb2 import ImgUnit
from tools.gpt_tools.mmu.media_common_result_status_pb2 import ResultCode

import uuid
import time

class GPT4oClient:
    def __init__(self, biz='luoxinchen_b585bcc7_gpt-4o-2024-08-06', timeout=120):
        client_option = ClientOption(
            biz_def='mmu',
            grpc_service_name='mmu-chat-gpt-service',
            grpc_stub_class=MmuChatGptServiceStub,
        )
        self.grpc_client = GrpcClient(client_option)
        self.biz = biz
        self.timeout = timeout
    
    def chat(self, prompt, images):
        answer = None
        try:
            sess_id = str(uuid.uuid4())
            request = MmuChatGptRequest(biz=self.biz)
            request.session_id = sess_id
            request.req_id = sess_id
            request.query = prompt
            # request.config['paygo_only'] = 'True'

            for image_data in images:
                img = ImgUnit(image = image_data)
                request.img.append(img)
            for i in range(10):
                resp = self.grpc_client.Chat(request, timeout=self.timeout)
                # print(resp)
                if resp.status.code == ResultCode.SUCESS:
                    answer = resp.answer
                    break
                else:
                    print(f"gpt4o rpc 失败，reponse={resp}, retry {i} times, sleep 30s.")
                    time.sleep(30)
                    continue
        except Exception as e:
            print('发生异常, err: %s', e)
        return answer


if __name__ == "__main__":
    
    gpt4o_client = GPT4oClient()

    prompt = "请针对这张图片生成html代码，要求尽可能还原图片中的格式和内容，可以忽略图中的条形码。注意，只输出html代码，不要包含其他文字。"
    images = []
    with open("/llm_reco_ssd/zhangzixing/recipe.jpeg", "rb") as fp:
        img_data = fp.read()
        images.append(img_data)

    answer = gpt4o_client.chat(prompt, images)
    print(answer)