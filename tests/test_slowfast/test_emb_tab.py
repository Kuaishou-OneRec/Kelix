from PIL import Image, ImageDraw
from PIL import Image
import torch
import sys
import os
os.environ["nosp"] = '1'
sys.path.append("./recovlm/models")
from keye_vitrope_slowfast_v2.processing_keye import KeyeProcessor
from keye_vitrope_slowfast_v2.keye_vl_utils import process_vision_info
# /llm_reco/lingzhixin/recovlm_qw0510/recovlm/recovlm/data/datasets.py
from recovlm.data.datasets import get_rope_index_slowfast
from recovlm.utils.ds_utils import print_input_info

'''
inputs["position_ids"] = get_rope_index_slowfast(
          input_ids = inputs["input_ids"],
          image_grid_thw=inputs.get("image_grid_thw", None),
          video_grid_thw=inputs.get("video_grid_thw", None),
          fast_video_grid_thw=inputs.get("fast_video_grid_thw", None),
          image_token_id=self.image_token_id,
          video_token_id=self.video_token_id,
          fast_video_token_id=self.fast_video_token_id,
          spatial_merge_size=self.spatial_merge_size,
          vision_start_token_id=self.vision_start_token_id,
      )
'''

MODEL_DIR = PROCESSOR_DIR = "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.9.7/Stage3/8b/1d_vs_3d_rope/rope1d_0.3.1014/step500/global_step500/converted/"
processor = KeyeProcessor.from_pretrained(PROCESSOR_DIR, use_fast=True, local_files_only=False, trust_remote_code=True)
messages = [
        {"role": "user", 
        "content": 
        [
            # {"type": "video", 
            # "video": "/llm_reco_ssd/caojiangxia/vllm/sample_videos/SampleVideo_1280x720_1mb.mp4", "video_total_pixels": 28*28*256}, 
            # {"type": "video", 
            # "video": ["/llm_reco_ssd/caojiangxia/vllm/test_image.png", "/llm_reco_ssd/caojiangxia/vllm/test_image.png"]},
            {"type": "video", 
            "video": "/llm_reco_ssd/caojiangxia/vllm/sample_videos/SampleVideo_1280x720_1mb.mp4", "video_total_pixels": 28*28*256}, 
            {"type": "text", 
            "text": "\\What is this?"}]},
    ]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
print_input_info(
    inputs, "inputsinputs"
)
inputs["position_ids"] = get_rope_index_slowfast(
          input_ids = inputs["input_ids"],
          image_grid_thw=inputs.get("image_grid_thw", None),
          video_grid_thw=inputs.get("video_grid_thw", None),
          fast_video_grid_thw=inputs.get("fast_video_grid_thw", None),
          image_token_id=151655,
          video_token_id=151656,
          fast_video_token_id=151678,
          spatial_merge_size=28,
          vision_start_token_id=151652,
      )
print(inputs)
exit()
'''
inputsinputs'input_ids':
inputsinputs  Tensor: shape=(7488, 3, 14, 14), dtype=torch.bfloat16, device=cpu, data=tensor([0.9922, 0.9922, 0.9922, 0.9922], dtype=torch.bfloat16)...tensor([0.9922, 0.9922, 0.9922, 0.9922], dtype=torch.bfloat16)
inputsinputs    stat0:  Full - mean: 0.934975, variance: 0.046155, max: 1.000000, min: -1.000000, non-zeros: 4402944
inputsinputs    stat1:  first half (2201472 elements) - mean: 0.899500, variance: 0.067203, max: 1.000000, min: -1.000000, non-zeros: 2201472
inputsinputs    stat2:  first 1000000 elements (magnitude-based) - mean: 0.938488, variance: 0.061735, max: 1.000000, min: -1.000000, non-zeros: 1000000
inputsinputs    stat3:  first 100000 elements (1/10 of magnitude-based) - mean: 0.992188, variance: 0.000000, max: 0.992188, min: 0.992188, non-zeros: 100000
inputsinputs'image_grid_thw':
inputsinputs  Tensor: shape=(1, 3), dtype=torch.int64, device=cpu, data=tensor([ 1, 96, 78])...tensor([ 1, 96, 78])
inputsinputs    stat0:  Full - mean: 58.333332, variance: 1697.555542, max: 96.000000, min: 1.000000, non-zeros: 3
inputsinputs    stat1:  first half (2 elements) - mean: 48.500000, variance: 2256.250000, max: 96.000000, min: 1.000000, non-zeros: 2
inputsinputs    stat2:  first 1 elements (magnitude-based) - mean: 1.000000, variance: 0.000000, max: 1.000000, min: 1.000000, non-zeros: 1
inputsinputs    stat3:  no elements (1/10 of magnitude-based <= 0) - mean: nan, variance: nan, max: nan, min: nan, non-zeros: 0
inputsinputs'loss_mask':
inputsinputs  Tensor: shape=(1, 4304), dtype=torch.int64, device=cpu, data=tensor([151644,   8948,    198,   2610])...tensor([  1773, 151645,    198, 151643])
inputsinputs    stat0:  Full - mean: 149221.796875, variance: 263340624.000000, max: 151655.000000, min: 13.000000, non-zeros: 4304
inputsinputs    stat1:  first half (2152 elements) - mean: 150899.593750, variance: 111227776.000000, max: 151655.000000, min: 13.000000, non-zeros: 2152
inputsinputs    stat2:  first 1000 elements (magnitude-based) - mean: 150029.375000, variance: 237947536.000000, max: 151655.000000, min: 13.000000, non-zeros: 1000
inputsinputs    stat3:  first 100 elements (1/10 of magnitude-based) - mean: 135398.828125, variance: 2141638656.000000, max: 151655.000000, min: 13.000000, non-zeros: 100
inputsinputs'attention_mask':
inputsinputs  Tensor: shape=(112860, 3, 14, 14), dtype=torch.bfloat16, device=cpu, data=tensor([0.6953, 0.6953, 0.6953, 0.7031], dtype=torch.bfloat16)...tensor([-0.1138, -0.1138, -0.1216, -0.2793], dtype=torch.bfloat16)
inputsinputs    stat0:  Full - mean: -0.062706, variance: 0.302617, max: 1.000000, min: -1.000000, non-zeros: 66361680
inputsinputs    stat1:  first half (33180840 elements) - mean: -0.078084, variance: 0.316160, max: 1.000000, min: -1.000000, non-zeros: 33180840
inputsinputs    stat2:  first 10000000 elements (magnitude-based) - mean: 0.011265, variance: 0.341314, max: 1.000000, min: -1.000000, non-zeros: 10000000
inputsinputs    stat3:  first 1000000 elements (1/10 of magnitude-based) - mean: 0.325201, variance: 0.337852, max: 1.000000, min: -1.000000, non-zeros: 1000000
inputsinputs'fast_video_grid_thw':
inputsinputs  Tensor: shape=(1, 2120), dtype=torch.int64, device=cpu, data=tensor([0, 0, 0, 0])...tensor([1, 1, 1, 0])
inputsinputs    stat0:  Full - mean: 0.082547, variance: 0.075733, max: 1.000000, min: 0.000000, non-zeros: 175
inputsinputs    stat1:  first half (1060 elements) - mean: 0.000000, variance: 0.000000, max: 0.000000, min: 0.000000, non-zeros: 0
inputsinputs    stat2:  first 1000 elements (magnitude-based) - mean: 0.000000, variance: 0.000000, max: 0.000000, min: 0.000000, non-zeros: 0
inputsinputs    stat3:  first 100 elements (1/10 of magnitude-based) - mean: 0.000000, variance: 0.000000, max: 0.000000, min: 0.000000, non-zeros: 0
inputsinputs  Tensor: shape=(135, 3), dtype=torch.int64, device=cpu, data=tensor([ 1, 22, 38,  1])...tensor([38,  1, 22, 38])
inputsinputs    stat0:  Full - mean: 20.333334, variance: 229.555557, max: 38.000000, min: 1.000000, non-zeros: 405
inputsinputs    stat1:  first half (203 elements) - mean: 20.246305, variance: 229.141312, max: 38.000000, min: 1.000000, non-zeros: 203
inputsinputs    stat2:  first 100 elements (magnitude-based) - mean: 20.139999, variance: 230.960403, max: 38.000000, min: 1.000000, non-zeros: 100
inputsinputs    stat3:  first 10 elements (1/10 of magnitude-based) - mean: 18.400000, variance: 240.240005, max: 38.000000, min: 1.000000, non-zeros: 10
'''
import torch

def process_pos_ids(pos_ids, input_ids):
    """
    处理positional id，基于外部提供的图像标记区分图片/非图片token
    
    规则：
    - 非图片token（is_image_token=False）：t/h/w均为0
    - 图片token（is_image_token=True）：
      - t：每张图片内部从1开始递增
      - h：每张图片内部从1开始递增（行坐标）
      - w：每张图片内部从1开始递增（列坐标）
    """
    # 提取t/h/w维度（形状：[N]）
    t = pos_ids[0, 0]
    h = pos_ids[1, 0]
    w = pos_ids[2, 0]
    device = t.device
    N = t.numel()  # 总token数

    is_image_token = (input_ids == fast_vid_pad) | (input_ids == vid_pad) | (input_ids == image_pad)
    is_image_token = is_image_token[0]

    # 转换图像标记为张量
    is_image = torch.tensor(is_image_token, dtype=torch.bool, device=device)
    
    # 初始化结果张量为长整型（解决类型不匹配问题）
    new_t = torch.zeros(N, device=device, dtype=torch.long)
    new_h = torch.zeros(N, device=device, dtype=torch.long)
    new_w = torch.zeros(N, device=device, dtype=torch.long)
    
    # 处理图像token
    if is_image.any():
        # 获取图像token索引
        img_idx = torch.where(is_image)[0]  # 图像token位置
        
        # 计算图像分组（连续索引为同一图像）
        if len(img_idx) > 1:
            group_flags = (img_idx[1:] - img_idx[:-1] > 1)  # 新图像标记
            groups = torch.cumsum(torch.cat([torch.tensor([1], device=device), group_flags]), 0) - 1
        else:
            groups = torch.tensor([0], device=device) if len(img_idx) == 1 else torch.tensor([], device=device)
        
        # 按图像分组处理
        for g in torch.unique(groups):
            mask = (groups == g)
            indices = img_idx[mask]  # 当前图像的所有token索引
            k = len(indices)  # 当前图像的token数量
            
            # 处理t维度：1~k递增
            new_t[indices] = torch.arange(1, k+1, device=device)
            
            # 处理h维度：组内最小值为基准
            group_h = h[indices]
            new_h[indices] = group_h - group_h.min() + 1
            
            # 处理w维度：组内最小值为基准
            group_w = w[indices]
            new_w[indices] = group_w - group_w.min() + 1
    
    # 重组为原始形状[3,1,N]
    return torch.stack([
        new_t.unsqueeze(0),
        new_h.unsqueeze(0),
        new_w.unsqueeze(0)
    ], dim=0)

# 测试用例
if __name__ == "__main__":
    pass


# 输入是
pos_ids = torch.tensor([
        [  # t维度（时序ID，连续递增）
            [0, 1, 2,          # t1, t2, t3
             3,3,3,3,3,3,3,3, # 第一个图像(4×2，8个patch)
             11,12,            # t4, t5
             13,13,13,13,13,13,# 第二个图像(3×2，6个patch)
             19,               # t6
             20,20,20,20,
             20,20,20,20,
             28,29
            ]
        ],
        [  # h维度（图像h = 前序最大t+1 + 行索引）
            [0, 1, 2,          # t1-t3：h=t
             3,3,3,3,4,4,4,4, # 第一个图像：行0→3，行1→4（前序最大t=2）
             11,12,            # t4-t5：h=t
             13,13,13,14,14,14,# 第二个图像：行0→13，行1→14（前序最大t=12）
             19,               # t6：h=t
             20,21,22,23,
             20,21,22,23,
             28,29
            ]
        ],
        [  # w维度（图像w = 前序最大t+1 + 列索引）
            [0, 1, 2,          # t1-t3：w=t
             3,4,5,6,3,4,5,6, # 第一个图像：列0-3→3-6（前序最大t=2）
             11,12,            # t4-t5：w=t
             13,14,15,13,14,15,# 第二个图像：列0-2→13-15（前序最大t=12）
             19,               # t6：w=t
             20,20,20,20,
             21,21,21,21,
             28,29
            ]
        ]
    ]) 
#以及一个bool类型的
is_image_token=[0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0]
#以及一个bool类型的
is_video_token=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0]

