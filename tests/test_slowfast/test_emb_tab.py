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
            {"type": "video", 
            "video": "/llm_reco_ssd/caojiangxia/vllm/sample_videos/SampleVideo_1280x720_1mb.mp4", "video_total_pixels": 28*28*128}, 
            #{"type": "video", 
            #"video": ["/llm_reco_ssd/caojiangxia/vllm/test_image.png", "/llm_reco_ssd/caojiangxia/vllm/test_image.png"]},
            {"type": "text", 
            "text": "What?"}]},]
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
    inputs, 
    # "inputsinputs"
)
inputs["position_ids"] = get_rope_index_slowfast(
          input_ids = inputs["input_ids"],
          image_grid_thw=inputs.get("image_grid_thw", None),
          video_grid_thw=inputs.get("video_grid_thw", None),
          fast_video_grid_thw=inputs.get("fast_video_grid_thw", None),
          image_token_id=151655,
          video_token_id=151656,
          fast_video_token_id=151678,
          spatial_merge_size=2,
          vision_start_token_id=151652,
      )

'''
inputsinputsDict: keys=7
inputsinputs'input_ids':
inputsinputs  Tensor: shape=(1, 4121), dtype=torch.int64, device=cpu, data=tensor([151644,   8948,    198,   2610])...tensor([    13, 151645,    198, 151643])
inputsinputs    stat0:  Full - mean: 143933.843750, variance: 1085754496.000000, max: 151680.000000, min: 13.000000, non-zeros: 4121
inputsinputs    stat1:  first half (2061 elements) - mean: 147785.687500, variance: 570349824.000000, max: 151680.000000, min: 13.000000, non-zeros: 2061
inputsinputs    stat2:  first 1000 elements (magnitude-based) - mean: 146853.984375, variance: 700425984.000000, max: 151680.000000, min: 13.000000, non-zeros: 1000
inputsinputs    stat3:  first 100 elements (1/10 of magnitude-based) - mean: 130850.640625, variance: 2662927104.000000, max: 151677.000000, min: 13.000000, non-zeros: 100
inputsinputs'attention_mask':
inputsinputs  Tensor: shape=(1, 4121), dtype=torch.int64, device=cpu, data=tensor([1, 1, 1, 1])...tensor([1, 1, 1, 1])
inputsinputs    stat0:  Full - mean: 1.000000, variance: 0.000000, max: 1.000000, min: 1.000000, non-zeros: 4121
inputsinputs    stat1:  first half (2061 elements) - mean: 1.000000, variance: 0.000000, max: 1.000000, min: 1.000000, non-zeros: 2061
inputsinputs    stat2:  first 1000 elements (magnitude-based) - mean: 1.000000, variance: 0.000000, max: 1.000000, min: 1.000000, non-zeros: 1000
inputsinputs    stat3:  first 100 elements (1/10 of magnitude-based) - mean: 1.000000, variance: 0.000000, max: 1.000000, min: 1.000000, non-zeros: 100
inputsinputs'pixel_values_videos':
inputsinputs  Tensor: shape=(9568, 3, 14, 14), dtype=torch.bfloat16, device=cpu, data=tensor([0.6016, 0.6016, 0.6016, 0.6016], dtype=torch.bfloat16)...tensor([-0.2793, -0.2793, -0.2793, -0.2793], dtype=torch.bfloat16)
inputsinputs    stat0:  Full - mean: 0.007288, variance: 0.176313, max: 1.000000, min: -1.000000, non-zeros: 5625984
inputsinputs    stat1:  first half (2812992 elements) - mean: 0.031437, variance: 0.173733, max: 1.000000, min: -1.000000, non-zeros: 2812992
inputsinputs    stat2:  first 1000000 elements (magnitude-based) - mean: 0.027855, variance: 0.156929, max: 1.000000, min: -1.000000, non-zeros: 1000000
inputsinputs    stat3:  first 100000 elements (1/10 of magnitude-based) - mean: 0.247987, variance: 0.060637, max: 1.000000, min: -0.648438, non-zeros: 100000
inputsinputs'video_grid_thw':
inputsinputs  Tensor: shape=(8, 3), dtype=torch.int64, device=cpu, data=tensor([ 1, 46, 26,  1])...tensor([26,  1, 46, 26])
inputsinputs    stat0:  Full - mean: 24.333334, variance: 338.888885, max: 46.000000, min: 1.000000, non-zeros: 24
inputsinputs    stat1:  first half (12 elements) - mean: 24.333334, variance: 338.888885, max: 46.000000, min: 1.000000, non-zeros: 12
inputsinputs    stat2:  first 10 elements (magnitude-based) - mean: 22.000000, variance: 354.000000, max: 46.000000, min: 1.000000, non-zeros: 10
inputsinputs    stat3:  first 1 elements (1/10 of magnitude-based) - mean: 1.000000, variance: 0.000000, max: 1.000000, min: 1.000000, non-zeros: 1
inputsinputs'fast_pixel_values_videos':
inputsinputs  Tensor: shape=(5712, 3, 14, 14), dtype=torch.bfloat16, device=cpu, data=tensor([-0.1138, -0.1138, -0.1138, -0.1138], dtype=torch.bfloat16)...tensor([-0.3340, -0.3340, -0.3340, -0.3262], dtype=torch.bfloat16)
inputsinputs    stat0:  Full - mean: -0.016213, variance: 0.183880, max: 1.000000, min: -1.000000, non-zeros: 3358656
inputsinputs    stat1:  first half (1679328 elements) - mean: 0.005986, variance: 0.186451, max: 1.000000, min: -1.000000, non-zeros: 1679328
inputsinputs    stat2:  first 1000000 elements (magnitude-based) - mean: -0.015328, variance: 0.194884, max: 1.000000, min: -1.000000, non-zeros: 1000000
inputsinputs    stat3:  first 100000 elements (1/10 of magnitude-based) - mean: 0.040561, variance: 0.123877, max: 1.000000, min: -1.000000, non-zeros: 100000
inputsinputs'fast_video_grid_thw':
inputsinputs  Tensor: shape=(17, 3), dtype=torch.int64, device=cpu, data=tensor([ 1, 24, 14,  1])...tensor([14,  1, 24, 14])
inputsinputs    stat0:  Full - mean: 13.000000, variance: 88.666664, max: 24.000000, min: 1.000000, non-zeros: 51
inputsinputs    stat1:  first half (26 elements) - mean: 12.961538, variance: 92.036980, max: 24.000000, min: 1.000000, non-zeros: 26
inputsinputs    stat2:  first 10 elements (magnitude-based) - mean: 11.800000, variance: 92.760002, max: 24.000000, min: 1.000000, non-zeros: 10
inputsinputs    stat3:  first 1 elements (1/10 of magnitude-based) - mean: 1.000000, variance: 0.000000, max: 1.000000, min: 1.000000, non-zeros: 1
inputsinputs'loss_mask':
inputsinputs  Tensor: shape=(1, 4121), dtype=torch.int64, device=cpu, data=tensor([0, 0, 0, 0])...tensor([1, 1, 1, 0])
inputsinputs    stat0:  Full - mean: 0.014317, variance: 0.014112, max: 1.000000, min: 0.000000, non-zeros: 59
inputsinputs    stat1:  first half (2061 elements) - mean: 0.000000, variance: 0.000000, max: 0.000000, min: 0.000000, non-zeros: 0
inputsinputs    stat2:  first 1000 elements (magnitude-based) - mean: 0.000000, variance: 0.000000, max: 0.000000, min: 0.000000, non-zeros: 0
inputsinputs    stat3:  first 100 elements (1/10 of magnitude-based) - mean: 0.000000, variance: 0.000000, max: 0.000000, min: 0.000000, non-zeros: 0
'''
import torch

def process_pos_ids(pos_ids, input_ids):
    fast_vid_pad = 151678
    vid_pad = 151656
    image_pad = 151655
    is_image_token = (input_ids == fast_vid_pad) | (input_ids == vid_pad) | (input_ids == image_pad)
    is_image_token = is_image_token[0]

    # 提取t/h/w维度（形状：[N]）
    t = pos_ids[0, 0]
    h = pos_ids[1, 0]
    w = pos_ids[2, 0]
    device = t.device
    N = t.numel()  # 总token数
    
    # 转换图像标记为张量
    is_image = torch.tensor(is_image_token, dtype=torch.bool, device=device)
    
    # 初始化结果张量为长整型（解决类型不匹配问题）
    # new_t = torch.zeros(N, device=device, dtype=torch.long)
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
            # new_t[indices] = torch.arange(1, k+1, device=device)
            
            # 处理h维度：组内最小值为基准
            group_h = h[indices]
            new_h[indices] = group_h - group_h.min() + 1
            
            # 处理w维度：组内最小值为基准
            group_w = w[indices]
            new_w[indices] = group_w - group_w.min() + 1
    
    # 重组为原始形状[3,1,N]
    return torch.stack([
        # new_t.unsqueeze(0),
        new_h.unsqueeze(0),
        new_w.unsqueeze(0)
    ], dim=0)

def generate_positional_id(thw):
    """
    将3x1xL的张量转换为1D positional_id矩阵
    
    参数:
        thw: 形状为(3, 1, L)的PyTorch张量
    
    返回:
        positional_id: 形状为(L,)的1D张量，包含连续的序列编号
    """
    # 检查输入形状是否正确
    assert thw.shape[0] == 3 and thw.shape[1] == 1, "输入必须是3x1xL的张量"
    
    # 取出第一位置并flatten
    seq = thw[0, 0, :].flatten()  # 形状变为(L,)
    
    # 识别子序列边界（假设以0为新子序列的开始）
    subsequence_starts = torch.where(seq == 0)[0].tolist()
    L = seq.numel()
    positional_id = torch.zeros_like(seq, dtype=torch.long)
    
    # 处理每个子序列
    for i, start in enumerate(subsequence_starts):
        # 确定当前子序列的结束位置
        if i < len(subsequence_starts) - 1:
            end = subsequence_starts[i + 1]
        else:
            end = L
        
        # 为当前子序列生成连续编号
        subsequence_length = end - start
        positional_id[start:end] = torch.arange(subsequence_length, dtype=torch.long)
    
    return positional_id

# 测试用例
if __name__ == "__main__":
    pos_ids = inputs["position_ids"]
    print("pos_ids=", pos_ids)
    print("process_pos_ids", process_pos_ids(pos_ids, inputs["input_ids"]))
    generated = generate_positional_id(pos_ids).to(pos_ids)[None, :]
    print("generated 1d rope pos id=", generated)
    exit()


'''
pos_ids(rope3d thw) = tensor([[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
          19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
          19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
          19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 29, 30, 31, 32, 33, 33,
          33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
          33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
          33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
          33, 33, 33, 33, 33, 33, 33, 43, 44, 45, 46, 47]],

        [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20,
          20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22,
          22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23,
          23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 29, 30, 31, 32, 33, 33,
          33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34,
          34, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36,
          36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38,
          38, 38, 38, 38, 38, 38, 38, 43, 44, 45, 46, 47]],

        [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 19, 20, 21, 22, 23,
          24, 25, 26, 27, 28, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 19, 20,
          21, 22, 23, 24, 25, 26, 27, 28, 19, 20, 21, 22, 23, 24, 25, 26, 27,
          28, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
          35, 36, 37, 38, 39, 40, 41, 42, 33, 34, 35, 36, 37, 38, 39, 40, 41,
          42, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 33, 34, 35, 36, 37, 38,
          39, 40, 41, 42, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 33, 34, 35,
          36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]])

从pos_ids(rope3d thw)转换为process_pos_ids(learnable hw) 
tensor([[[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,
           2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,
           4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,
           5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  0,  0,  0,  0,  1,  1,
           1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,
           2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,
           4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,
           6,  6,  6,  6,  6,  6,  6,  0,  0,  0,  0,  0]],

        [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  1,  2,  3,  4,  5,
           6,  7,  8,  9, 10,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  1,  2,
           3,  4,  5,  6,  7,  8,  9, 10,  1,  2,  3,  4,  5,  6,  7,  8,  9,
          10,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  0,  0,  0,  0,  1,  2,
           3,  4,  5,  6,  7,  8,  9, 10,  1,  2,  3,  4,  5,  6,  7,  8,  9,
          10,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  1,  2,  3,  4,  5,  6,
           7,  8,  9, 10,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  1,  2,  3,
           4,  5,  6,  7,  8,  9, 10,  0,  0,  0,  0,  0]]])

generated 1d rope pos id= 
tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
          14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
          28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
          42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
          56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
          70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
          84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
          98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
         112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
         126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
         140, 141, 142, 143, 144, 145, 146, 147]])
'''



# # 输入是
# pos_ids = torch.tensor([
#         [  # t维度（时序ID，连续递增）
#             [0, 1, 2,          # t1, t2, t3
#              3,3,3,3,3,3,3,3, # 第一个图像(4×2，8个patch)
#              11,12,            # t4, t5
#              13,13,13,13,13,13,# 第二个图像(3×2，6个patch)
#              19,               # t6
#              20,20,20,20,
#              20,20,20,20,
#              28,29
#             ]
#         ],
#         [  # h维度（图像h = 前序最大t+1 + 行索引）
#             [0, 1, 2,          # t1-t3：h=t
#              3,3,3,3,4,4,4,4, # 第一个图像：行0→3，行1→4（前序最大t=2）
#              11,12,            # t4-t5：h=t
#              13,13,13,14,14,14,# 第二个图像：行0→13，行1→14（前序最大t=12）
#              19,               # t6：h=t
#              20,21,22,23,
#              20,21,22,23,
#              28,29
#             ]
#         ],
#         [  # w维度（图像w = 前序最大t+1 + 列索引）
#             [0, 1, 2,          # t1-t3：w=t
#              3,4,5,6,3,4,5,6, # 第一个图像：列0-3→3-6（前序最大t=2）
#              11,12,            # t4-t5：w=t
#              13,14,15,13,14,15,# 第二个图像：列0-2→13-15（前序最大t=12）
#              19,               # t6：w=t
#              20,20,20,20,
#              21,21,21,21,
#              28,29
#             ]
#         ]
#     ]) 
# #以及一个bool类型的
# is_image_token=[0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0]
# #以及一个bool类型的
# is_video_token=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0]

