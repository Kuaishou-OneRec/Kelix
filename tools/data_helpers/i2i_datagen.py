import csv

# 定义文件路径
triplets_path = "/llm_reco/maosiyang/dataset/i2i/triplets"
ocr_asr_path = "/llm_reco/maosiyang/dataset/i2i/ocr_asr_data_1c89f9c17f1546d295b9c903ab91dab9"
output_csv = "/llm_reco/maosiyang/dataset/i2i/triplets_imf.csv"

# 第一步：读取ocr_asr_data并建立photoId到数据的映射
ocr_asr_map = {}
with open(ocr_asr_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 6:  # 确保数据格式正确
            photo_id = parts[0]
            # 使用切片获取后续字段（即使字段数量超过6也只取前5个）
            fields = parts[1:6] + ['']*(5-len(parts[1:6]))  # 补全缺失字段
            ocr_asr_map[photo_id] = {
                'caption': fields[0],
                'title': fields[1],
                'text': fields[2],
                'ocr': fields[3],
                'asr': fields[4]
            }

# 第二步：处理triplets并生成CSV
with open(triplets_path, 'r', encoding='utf-8') as triplets, \
     open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

    writer = csv.writer(outfile)
    
    # 写入CSV头部
    header = [
        'src_pid', 'src_caption', 'src_title', 'src_text', 'src_ocr', 'src_asr',
        'sim_pid', 'sim_caption', 'sim_title', 'sim_text', 'sim_ocr', 'sim_asr',
        'neg_pid', 'neg_caption', 'neg_title', 'neg_text', 'neg_ocr', 'neg_asr'
    ]
    writer.writerow(header)
    
    # 处理每一行三元组
    for line in triplets:
        pids = line.strip().split('\t')
        if len(pids) != 3:
            continue  # 跳过格式错误行
            
        src_pid, sim_pid, neg_pid = pids
        row = []
        
        # 为每个pid获取对应字段
        for pid in [src_pid, sim_pid, neg_pid]:
            data = ocr_asr_map.get(pid, {})
            row.extend([
                pid,
                data.get('caption', ''),
                data.get('title', ''),
                data.get('text', ''),
                data.get('ocr', ''),
                data.get('asr', '')
            ])
        
        writer.writerow(row)

print(f"CSV文件已生成：{output_csv}")