import numpy as np
import time
import math
import bisect
import sys
import torch.distributed as dist
from typing import List

def sampling(input_ids_len, target_size=200, n_bins=20):
    input_ids_len = np.array(input_ids_len, dtype=np.int64)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bounds = np.percentile(input_ids_len, percentiles)
    
    bin_counts = np.histogram(input_ids_len, bins=bounds)[0]
    sample_counts = (bin_counts / len(input_ids_len) * target_size).astype(int)
    
    sample_counts[np.argmax(sample_counts)] += target_size - sample_counts.sum()
    
    sampled = []
    for i in range(n_bins):
        in_bin = input_ids_len[(input_ids_len >= bounds[i]) & (input_ids_len < bounds[i+1])]
        sampled.extend(np.random.choice(in_bin, sample_counts[i], replace=False))
    
    return np.array(sampled).tolist()


def greedy_subsets_nearst_sum(nums, N):
    key_fn = lambda x : x[0]
    ordered = sorted([(num, idx) for idx, num in enumerate(nums)], key=key_fn)
    result = []
    for i in reversed(range(len(nums))):
        current = []
        cur_sum = 0
        j = i
        while j >= 0:
            cur_sum += ordered[j][0]
            if cur_sum > N:
                break
            else:
                current = [ordered[j]] + current
                j = bisect.bisect_right(ordered, N - cur_sum, 0, j, key=key_fn) - 1
        result.append(current)
    sorted_result = sorted(result, key=lambda x: -sum(v[0] for v in x))
    result_index = []
    for res in sorted_result:
        result_index.append([v[1] for v in res])
    return result_index
    

def greedy_subsets_without_replacement(nums, N, m):
    ordered = sorted([(num, idx) for idx, num in enumerate(nums)], key=lambda x: x[0])
    values = [x[0] for x in ordered]
    result = []
    used_indices = set()
    
    for i in reversed(range(len(nums))):
        if ordered[i][1] in used_indices:
            continue

        current = []
        cur_sum = 0
        j = i
        
        while j >= 0:
            num, idx = ordered[j]
            if idx in used_indices or cur_sum + num > N:
                j -= 1
                continue
                
            current.append((num, idx))
            cur_sum += num
            used_indices.add(idx)
            
            remaining = N - cur_sum
            if remaining <= 0:
                break
                
            pos = bisect.bisect_right(values, remaining, 0, j) - 1
            j = pos
            while j >= 0 and (ordered[j][1] in used_indices or ordered[j][0] > remaining):
                j -= 1
                
        if current:
            result.append(current)
    
    # sorted_result = sorted(result, key=lambda x: -sum(v[0] for v in x))
    sorted_result = sorted(result, key=lambda x: -m.llm_flops([v[0] for v in x]))
    result_index = [[v[1] for v in res] for res in sorted_result]
    return result_index


class ModelFlopsBase(object):
    def __init__(self, flops_range, **kwargs):
        self._flops_range = flops_range
        assert self._flops_range is not None

    @property
    def flops_range(self):
        return self._flops_range

    def llm_flops(self, seq_list: List[int]) -> float:
        raise NotImplemented("llm_flops")
    
    def vit_flops(self, image_list: List[int]) -> float:
        raise NotImplemented("vit_flops")

    def calculate_llm_flops_range(self, maxlen, ratio, max_sample_num):
        assert max_sample_num > 0
        max_f = self.llm_flops([maxlen])
        min_len_per_sample = int(maxlen // max_sample_num)
        min_f = self.llm_flops([min_len_per_sample] * max_sample_num + [maxlen % max_sample_num])
        flops = []
        flops.append(max_f)
        assert ratio < 1 and ratio > 0
        flops_size = int(1 // ratio)
        for i in range(flops_size):
            next_f = int(flops[-1] * (1 - ratio))
            if next_f < min_f:
                break
            flops.append(next_f)
        return sorted(flops[1:])


class InternVLChatModelFlops(ModelFlopsBase):
    def __init__(self, **kwargs):
        max_len = kwargs["max_length"]
        max_flops = self.llm_flops([max_len])
        max_sample = kwargs.get("max_sample_num", 1000)
        diff_ratio = kwargs.get("diff_ratio", 0.05)
        flops_range = self.calculate_llm_flops_range(max_len, diff_ratio, max_sample)
        print(f"InternVLChatModelFlops range: {flops_range}")
        super(InternVLChatModelFlops, self).__init__(flops_range)
    
    def llm_flops(self, seq_list: List[int]) -> float:
        h = 1536
        intermediate_size = 8960
        attention = 0
        seq_sum = 0
        for s in seq_list:
            attention += 2 * h * s * s
            seq_sum += s
        return (8 * seq_sum * h * h + attention + 6 * seq_sum * h * intermediate_size)  * 28 * 3 / 1e12

    def vit_flops(self, image_list: List[int]) -> float:
        h = 1024
        attention = 0
        seq_sum = 0
        for s in image_list:
            attention += 4 * s * h * 1024 * 1024
            seq_sum += s * 1024
        return (24 * seq_sum * h * h + attention) * 24 * 3 / 1e12


class Qwen3SiglipModelFlops(ModelFlopsBase):
    def __init__(self, **kwargs):
        max_len = kwargs["max_length"]
        max_flops = self.llm_flops([max_len])
        max_sample = kwargs.get("max_sample_num", 1000)
        diff_ratio = kwargs.get("diff_ratio", 0.05)
        flops_range = self.calculate_llm_flops_range(max_len, diff_ratio, max_sample)
        print(f"Qwen3SiglipModelFlops range: {flops_range}")
        self.kwargs = kwargs
        super(Qwen3SiglipModelFlops, self).__init__(flops_range)
    
    def llm_flops(self, seq_list: List[int]) -> float:
        h = 4096
        intermediate_size = 12288
        attention = 0
        seq_sum = 0
        for s in seq_list:
            attention += 2 * h * s * s
            seq_sum += s
        flops_q_o = 4 * seq_sum * h * h
        flops_k_v = 4 * seq_sum * h * 1024
        flops_ffn = 3 * 2 * seq_sum * h * intermediate_size
        return (flops_q_o + flops_k_v + flops_ffn + attention)  * 36 * 3 / 1e12

    def vit_flops(self, image_list: List[int]) -> float:
        h = 1152
        intermediate_size = 4304
        attention = 0
        seq_sum = 0
        for s in image_list:
            attention += 4 * h * s * s
            seq_sum += s
        flops_qkvo = 4 * 2 * seq_sum * h * h
        flops_ffn = 2 * 2 * seq_sum * h * intermediate_size
        return (flops_qkvo + flops_ffn + attention) * 27 * 3 / 1e12


from tools.mfu.flops_counter import calculate_llm_flops_from_config, calculate_vit_flops_from_config
class CustomModelFlops(ModelFlopsBase):
    def __init__(self, base_model_config, **kwargs):
        self.base_model_config = base_model_config

        import os
        import json
        with open(os.path.join(self.base_model_config), "r") as fp:
            config = json.load(fp)
            self.arch = config["architectures"][0]

        print(f"self.arch={self.arch}")
        max_len = kwargs["max_length"]
        max_flops = self.llm_flops([max_len])
        max_sample = kwargs.get("max_sample_num", 1000)
        diff_ratio = kwargs.get("diff_ratio", 0.05)
        flops_range = self.calculate_llm_flops_range(max_len, diff_ratio, max_sample)
        print(f"CustomModelFlops range: {flops_range}")
        self.kwargs = kwargs
        super(CustomModelFlops, self).__init__(flops_range)

    def llm_flops(self, seq_list: List[int]) -> float:
        return calculate_llm_flops_from_config(self.base_model_config, seq_list, None)['total_flops'] / 1e12

    def vit_flops(self, image_list: List[int]) -> float:
        return calculate_vit_flops_from_config(self.base_model_config, image_list, None)['total_flops'] / 1e12


def flops_diff(flops1, flops2):
    return (flops1[0] - flops2[0]) ** 2 + math.fabs(flops1[1] - flops2[1])


def greedy_find_by_diff(current_flops, candidates):
    key_fn = lambda x: x[0]
    curmax = max(current_flops, key=key_fn)
    curmin = min(current_flops, key=key_fn)
    diff = sys.maxsize
    found = None
    for c in candidates:
        newmax = max(curmax, c, key=key_fn)
        newmin = min(curmin, c, key=key_fn)
        newdiff = flops_diff(newmax, newmin)
        if newdiff < diff:
            diff = newdiff
            found = c
    assert found is not None, f"{current_flops}, {candidates}"
    return found



def select_by_flops(all_flops, rank):
    flops_pair = []
    for flops in all_flops:
        sorted_pair = sorted(flops, key=lambda x : x[0])
        flops_pair.append(sorted_pair)

    def find_best(arr, target):
        pos = bisect.bisect_left(arr, target[0], key=lambda x : x[0])
        best = None
        diff = sys.maxsize
        for i in range(pos - 1, pos + 1):
            if i >= 0 and i < len(arr):
                d = flops_diff(arr[i], target)
                if d < diff:
                    best = arr[i]
                    diff = d
        return best 

    current = flops_pair[rank]
    result = None
    min_diff = sys.maxsize
    for p in current:
        local = []
        curflops = []
        curflops.append(p)
        for i, flops in enumerate(flops_pair):
            if i == rank:
                local.append(p)
                continue
            select = greedy_find_by_diff(curflops, flops)
            curflops.append(select)
            local.append(select)
        local_diff = flops_diff(max(local, key=lambda x: x[0]), min(local, key=lambda x: x[0]))
        if local_diff < min_diff:
            min_diff = local_diff
            result = local
    return result


def find_global(flops_list):
    best = None
    min_diff = sys.maxsize 
    for flops in flops_list:
        local_diff = flops_diff(max(flops, key=lambda x: x[0]), min(flops, key=lambda x: x[0]))
        if local_diff < min_diff:
            best = flops
            min_diff = local_diff
    return best


def find_local_v1(nums, N):
    key_fn = lambda x : x[0]
    ordered = sorted([(num, idx) for idx, num in enumerate(nums)], key=key_fn)
    result = []
    j = len(nums) - 1
    cur_sum = 0
    while j >= 0:
        cur_sum += ordered[j][0]
        if cur_sum > N:
            break
        else:
            result = [ordered[j]] + result
            j = bisect.bisect_right(ordered, N - cur_sum, 0, j, key=key_fn) - 1
    result_index = [v[1] for v in result]
    return result_index


def calculate_transfer_scheme(numbers):
    n = len(numbers)
    total = sum(numbers)
    v = total // n
    
    receivers = [(i, v - num) for i, num in enumerate(numbers) if num < v]
    givers = [(i, num - v) for i, num in enumerate(numbers) if num > v]
    
    transfers = []
    giver_idx = 0
    
    for receiver_idx, needed in receivers:
        while needed > 0:
            giver_i, available = givers[giver_idx]
            transfer_amount = min(needed, available)
    
            transfers.append((giver_i, receiver_idx, transfer_amount))
    
            givers[giver_idx] = (giver_i, available - transfer_amount)
            needed -= transfer_amount
    
            if available - transfer_amount == 0:
                giver_idx += 1
    
    return v, transfers



def exchange_batch_info(samples, ds_list, m):
    assert len(ds_list) == len(samples)
    N = len(samples)
    input_len = [s["input_ids"].shape[-1] for s in samples]
    if isinstance(m, InternVLChatModelFlops) or (
            isinstance(m, CustomModelFlops) and m.arch == "InternVLChatModel"):
        image_len = [s["pixel_values"].size(0) for s in samples]
    elif isinstance(m, Qwen3SiglipModelFlops) or (
            isinstance(m, CustomModelFlops) and (
            m.arch == "Qwen3SiglipModel" or m.arch == "KeyeForConditionalGeneration" or m.arch == "Qwen3ForCausalLM")):
        print("maaaaaaaa")
        image_len = []
        for s in samples:
            if "image_grid_thw" not in s:
                continue
            thw = s["image_grid_thw"]
            lens = [(thw[i][1] * thw[i][2]).item() for i in range(thw.size(0))]
            image_len.extend(lens)
    f1 = m.llm_flops(input_len)
    f2 = m.vit_flops(image_len)
    info = [N, sum(input_len), sum(image_len), f1, f2] + ds_list
    global_info = [None] * dist.get_world_size()
    dist.all_gather_object(global_info, info)
    num_samples = sum(info[0] for info in global_info)
    num_input_ids = sum(info[1] for info in global_info)
    num_images = sum(info[2] for info in global_info)
    llm_flops_list = [info[3] for info in global_info]
    vit_flops_list = [info[4] for info in global_info]
    return (num_samples, num_input_ids, num_images, llm_flops_list, vit_flops_list)



def get_flops_model(model_type, **kwargs) -> ModelFlopsBase:
    if model_type == "InternVLChatModel":
        return InternVLChatModelFlops(**kwargs)
    elif model_type == "Qwen3SiglipForConditionalGeneration_navit":
        return Qwen3SiglipModelFlops(**kwargs)
    else:
        raise RuntimeError(f"Not supported flops computation for {model_type}")


if __name__ == '__main__':
    raw_data = [27, 28, 33, 38, 39, 40, 40, 42, 46, 46, 50, 53, 55, 58, 58, 58, 59, 59, 60, 61, 61, 64, 65, 65, 65, 69, 69, 70, 71, 73, 74, 75, 76, 85, 87, 87, 101, 108, 111, 111, 137, 139, 158, 160, 173, 183, 189, 200, 202, 203, 221, 230, 246, 262, 265, 267, 269, 269, 272, 276, 277, 278, 284, 285, 285, 289, 291, 298, 298, 299, 299, 299, 299, 300, 300, 300, 300, 300, 300, 301, 302, 302, 302, 303, 303, 303, 303, 303, 304, 304, 304, 304, 304, 304, 305, 305, 306, 306, 306, 306, 306, 306, 306, 306, 306, 307, 308, 308, 308, 308, 308, 308, 308, 309, 309, 309, 309, 309, 309, 310, 310, 310, 310, 310, 311, 311, 311, 311, 312, 312, 312, 313, 313, 313, 313, 313, 313, 314, 314, 315, 316, 317, 319, 319, 320, 320, 320, 321, 323, 324, 325, 327, 329, 334, 335, 338, 340, 340, 348, 349, 362, 368, 373, 385, 391, 393, 396, 401, 406, 409, 411, 425, 428, 431, 432, 432, 436, 438, 469, 482, 509, 511, 514, 517, 522, 528, 543, 543, 544, 545, 565, 569, 573, 577, 578, 592, 592, 610, 620, 628, 640, 642, 644, 646, 663, 668, 683, 697, 732, 741, 749, 761, 776, 794, 796, 799, 801, 803, 807, 808, 808, 811, 811, 811, 811, 812, 812, 812, 813, 814, 818, 818, 819, 821, 821, 821, 823, 823, 823, 825, 827, 827, 828, 830, 831, 838, 840, 841, 846, 847, 848, 849, 849, 851, 851, 854, 858, 866, 867, 888, 892, 895, 926, 931, 993, 1018, 1023, 1031, 1064, 1068, 1077, 1079, 1090, 1094, 1097, 1099, 1112, 1117, 1134, 1135, 1139, 1148, 1156, 1159, 1164, 1209, 1244, 1260, 1262, 1270, 1293, 1309, 1313, 1314, 1317, 1324, 1329, 1329, 1331, 1331, 1332, 1333, 1333, 1338, 1341, 1346, 1350, 1350, 1356, 1359, 1514, 1523, 1553, 1573, 1595, 1611, 1612, 1653, 1811, 1816, 1817, 1818, 1818, 1819, 1821, 1822, 1823, 1824, 1828, 1832, 1833, 1834, 1834, 1834, 1835, 1835, 1835, 1836, 1836, 1836, 1836, 1836, 1837, 1837, 1838, 1838, 1838, 1838, 1838, 1839, 1839, 1839, 1839, 1839, 1839, 1839, 1840, 1840, 1840, 1840, 1840, 1841, 1841, 1841, 1841, 1842, 1842, 1842, 1842, 1842, 1842, 1842, 1843, 1843, 1843, 1843, 1844, 1844, 1845, 1845, 1845, 1845, 1845, 1846, 1846, 1846, 1846, 1847, 1847, 1847, 1848, 1849, 1849, 1849, 1850, 1850, 1852, 1853, 1854, 1855, 1855, 1857, 1858, 1858, 1859, 1860, 1861, 1861, 1862, 1862, 1864, 1868, 1869, 1870, 1871, 1871, 1872, 1873, 1873, 1874, 1875, 1876, 1882, 1883, 1898, 1905, 1907, 1932, 1935, 1940, 1946, 1955, 1955, 1962, 1966, 1974, 1986, 1991, 1997, 2007, 2008, 2008, 2010, 2103, 2109, 2110, 2114, 2118, 2130, 2133, 2155, 2160, 2181, 2191, 2195, 2219, 2269, 2282, 2284, 2313, 2319, 2325, 2326, 2349, 2361, 2363, 2365, 2411, 2433, 2455, 2481, 2508, 2575, 2575, 2589, 2606, 2609, 2614, 2616, 2617, 2618, 2620, 2624, 2628, 2631, 2725, 2758, 2795, 2844, 2858, 2860, 2861, 2862, 2864, 2864, 2881, 2888, 2891, 2891, 2902, 2978, 3062, 3116, 3322, 3342, 3346, 3348, 3356, 3363, 3364, 3367, 3367, 3368, 3368, 3369, 3369, 3370, 3370, 3370, 3370, 3370, 3370, 3371, 3371, 3371, 3371, 3372, 3372, 3372, 3372, 3373, 3373, 3373, 3373, 3373, 3373, 3374, 3374, 3374, 3374, 3374, 3374, 3374, 3374, 3375, 3375, 3375, 3375, 3375, 3375, 3375, 3376, 3376, 3376, 3376, 3376, 3376, 3376, 3376, 3376, 3377, 3377, 3377, 3377, 3377, 3377, 3377, 3377, 3377, 3378, 3378, 3378, 3378, 3378, 3378, 3378, 3379, 3379, 3379, 3379, 3379, 3379, 3379, 3380, 3380, 3380, 3380, 3381, 3381, 3381, 3381, 3381, 3381, 3382, 3382, 3382, 3382, 3382, 3383, 3383, 3383, 3383, 3383, 3383, 3383, 3383, 3383, 3383, 3384, 3384, 3384, 3384, 3384, 3384, 3384, 3384, 3384, 3384, 3385, 3385, 3385, 3385, 3385, 3385, 3386, 3386, 3387, 3387, 3387, 3387, 3387, 3388, 3388, 3389, 3389, 3389, 3390, 3390, 3390, 3391, 3391, 3391, 3391, 3391, 3391, 3391, 3392, 3392, 3392, 3393, 3393, 3394, 3394, 3394, 3394, 3394, 3395, 3395, 3395, 3397, 3398, 3398, 3398, 3399, 3399, 3400, 3401, 3404, 3404, 3406, 3406, 3406, 3407, 3407, 3408, 3408, 3409, 3413, 3416, 3417, 3421, 3424, 3425, 3433, 3435, 3438, 3441, 3457, 3469, 3473, 3485, 3502, 3506, 3508, 3516, 3524, 3527, 3532, 3533, 3556, 3571, 3578, 3612, 3624, 3680, 3687, 3689, 3695, 3798, 3799, 3867, 3905, 3924, 4008, 4008, 4204, 4229, 4279, 4357, 4494, 4518, 4594, 4722, 4819, 4825, 4853, 5064, 5190, 5420, 5590, 5606, 5718, 5793, 6020, 6185, 6237, 6245, 6420, 6476, 6677, 6697, 6730, 6760, 6763, 6792, 6894, 6911, 6986, 7064, 7248, 7254, 7350, 7385, 7432, 7495, 7506, 7897, 8222, 8264, 8319, 8544, 8985, 9043, 9074, 9097, 9104, 9179, 9317, 9402, 9544, 9634, 9670, 9692, 9733, 10093, 10110, 10207, 10324, 10391, 10421, 10470, 10607, 10822, 10853, 10990, 11072, 11231, 11245, 11387, 11579, 11677, 11736, 11977, 12145, 12188, 12342, 12537, 12679, 12748, 12781, 12827, 12932, 12960, 13002, 13112, 13140, 13298, 13455, 13530, 13745, 13874, 13994, 14045, 14051, 14173, 14178, 14187, 14352, 14409, 14451, 14509, 14609, 14787, 14925, 15015, 15021, 15095, 15128, 15264, 15440, 15551, 15808, 15867, 16036, 16374, 16426, 16436, 16438, 16486, 16504, 16556, 16570, 16604, 16624, 16626, 16654, 16720, 16745, 16779, 16812, 16814, 16968, 16979, 16998, 17026, 17116, 17142, 17201, 17203, 17306, 17361, 17386, 17387, 17455, 17521, 17657, 17688, 17692, 17764, 17845, 17866, 17880, 17921, 17950, 17973, 17974, 17979, 18028, 18034, 18045, 18046, 18089, 18147, 18171, 18180, 18192, 18206, 18222, 18253, 18261, 18276, 18282, 18333, 18333, 18390, 18398, 18436, 18451, 18456, 18457, 18463, 18470, 18476, 18496, 18518, 18586, 18605, 18615, 18754, 18760, 18803, 18812, 18892, 18943, 18970, 19027, 19042, 19045, 19089, 19104, 19154, 19322, 19344, 19392, 19457, 19507, 19544, 19604, 19669, 19676, 19686, 19707, 19717, 19802, 19927, 19953, 20001, 20034, 20034, 20044, 20066, 20075, 20088, 20090, 20095, 20097, 20118, 20132, 20133, 20150, 20160, 20163, 20176, 20204, 20221, 20222, 20243, 20250, 20257, 20274, 20276, 20280, 20293, 20293, 20301, 20352, 20353, 20355, 20369, 20385, 20404, 20408, 20410, 20435, 20438, 20439, 20450, 20451, 20466, 20466, 20469, 20476, 20484, 20504, 20511, 20526, 20545, 20577, 20579, 20599, 20602, 20624, 20627, 20628, 20670, 20699, 20705, 20713, 20717, 20725]
    
    raw_image = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 , 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 15, 16, 20, 20, 21, 21, 21, 21, 21, 22, 23, 23, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 28, 28, 30, 31, 32, 32, 32, 33, 34, 34, 35, 35, 35, 35, 35, 35, 36, 36, 38, 39, 39, 39, 39, 39, 40, 40, 41, 42, 43, 46, 46, 46, 46, 46, 47, 48, 49, 49, 50, 50, 50, 51, 52, 53, 54, 54, 54, 56, 56, 57, 57, 57, 57, 57, 59, 60, 61, 61, 62, 62, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 66, 66, 66, 67, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 70, 70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 79, 79, 79, 80, 80, 80, 80]
    
    t1 = time.perf_counter()
    data = sampling(raw_data, 200)
    sample_index = [raw_data.index(v) for v in data]
    image = [raw_image[i] for i in sample_index]
    t2 = time.perf_counter()
    print(t2 - t1)
    print(sorted(data))
    
    
    t1 = time.perf_counter()
    found = greedy_subsets_nearst_sum(data, 21000)
    t2 = time.perf_counter()
    
    from prettytable import PrettyTable
    t = PrettyTable(['llm_seq', 'llm_sum', 'llm_flops', 'vit_seq', 'vit_sum', 'vit_flops'])
    print(found)
    
    m = InternVLChatModelFlops()
    for v in found:
        seq_len = [data[i] for i in v]
        image_len = [image[i] for i in v]
        t.add_row([seq_len, sum(seq_len), m.llm_flops(seq_len), image_len, sum(image_len), m.vit_flops(image_len)])
        print(seq_len, sum(seq_len), m.llm_flops(seq_len))
    t3 = time.perf_counter()
    print(t2 - t1, t3-t2, t)
    
