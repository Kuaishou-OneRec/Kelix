import pandas as pd
import json

# 读取Parquet文件
file_path = '/llm_reco/maosiyang/dataset/wenjuan/rank-0-5db9c286-fa6d-11ef-be0f-a400e2727258.parquet'
df = pd.read_parquet(file_path)

# 统计assistant的回答中包含“和源视频最相似的视频是【视频2】”的数量
count = 0
for row in df['messages']:
    messages = json.loads(row)
    for message in messages:
        if message['role'] == 'assistant':
            for content in message['content']:
                if content['type'] == 'text' and "【结果：不满意】" in content['text']:
                    count += 1

print(f"【结果：不满意】的回答数量: {count}") 