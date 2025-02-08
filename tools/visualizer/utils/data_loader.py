import json
import os
import pandas as pd
import pyarrow.parquet as pq
import glob
import random
import pyarrow.hdfs as hdfs

def load_data(data_path):
    """加载数据并确保保存目录存在"""
    data = json.load(open(data_path))
    return data

def read_parquet_with_nrows(data_path, nrows=None, shuffle=False):
    """读取parquet文件，支持行数限制和HDFS路径"""
    is_hdfs = data_path.startswith('viewfs://')
    
    if is_hdfs:
        fs = hdfs.connect()
        files = [data_path] if data_path.endswith('.parquet') else [f for f in fs.ls(data_path) if f.endswith('.parquet')]
    else:
        if os.path.isfile(data_path) and data_path.endswith('.parquet'):
            files = [data_path]
        elif os.path.isdir(data_path):
            files = glob.glob(os.path.join(data_path, '*.parquet'))
        else:
            files = [data_path]

    if shuffle:
        random.shuffle(files)
    
    dfs = []
    rows_read = 0
    
    for file in files:
        if nrows is not None and rows_read >= nrows:
            break
            
        table = pq.read_table(file)
        df = table.to_pandas()
        
        if len(df) > 0 and shuffle:
            df = df.sample(frac=1.0, random_state=None)
        
        if nrows is not None:
            remaining_rows = nrows - rows_read
            if len(df) > remaining_rows:
                df = df.iloc[:remaining_rows]
            
        dfs.append(df)
        rows_read += len(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame() 