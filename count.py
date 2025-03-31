import pandas as pd
import json

# 读取Parquet文件
file_path = '/llm_reco/maosiyang/dataset/i2i/rank-0-dad122ea-fcb7-11ef-a51a-a400e2727258.parquet'
df = pd.read_parquet(file_path)

# 统计assistant的回答中包含“和源视频最相似的视频是【视频2】”的数量
count = 0
allcount = 0
for row in df['messages']:
    messages = json.loads(row)
    for message in messages:
        if message['role'] == 'assistant':
            for content in message['content']:
                if content['type'] == 'text' and "和源视频最相似的视频是【视频2】" in content['text']:
                    count += 1
                allcount += 1
print(f"和源视频最相似的视频是【视频2】的回答数量: {count}") 
print(f"总回答数量: {allcount}")
print(f"和源视频最相似的视频是【视频1】的回答数量: {allcount - count}")