import pandas as pd
import json

# 读取Parquet文件
file_path = '/llm_reco_ssd/maosiyang/wenjuan_benchmark_dataset/rank-0-c089702c-faa0-11ef-a168-a400e2727258.parquet'
df = pd.read_parquet(file_path)

# 统计assistant的回答中包含“和源视频最相似的视频是【视频2】”的数量
count = 0
for index, row in df.iterrows():
    meta = json.loads(row)
    messages = json.loads(meta['messages'])
    for message in messages:
        if message['role'] == 'assistant':
            for content in message['content']:
                if content['type'] == 'text' and "和源视频最相似的视频是【视频2】" in content['text']:
                    count += 1

print(f"和源视频最相似的视频是【视频2】的回答数量: {count}")