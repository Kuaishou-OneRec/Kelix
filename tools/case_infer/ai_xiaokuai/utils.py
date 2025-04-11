from omegaconf import OmegaConf
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import base64
import random
import uuid
import json



def save_parquet(image_list, text_list, prompt_list, output_dir):
    out = []
    for images, text, prompt in zip(image_list, text_list, prompt_list):
        result = {
            "images": json.dumps(images),  # 保存图片路径列表
            "prompt": prompt,              # 保存提示词
            "response": text               # 保存模型回复
        }
        out.append(result)

    df = pd.DataFrame(out)  # 将列表转换为 Pandas DataFrame
    table = pa.Table.from_pandas(df)
    directory = os.path.dirname(output_dir)
    os.makedirs(directory, exist_ok=True)
    pq.write_table(table, output_dir)