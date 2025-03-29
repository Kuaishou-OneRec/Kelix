import pandas as pd
import pyarrow.parquet as pq
import json
from typing import List, Dict
from dataclasses import dataclass
from collections import defaultdict

from tqdm import tqdm

@dataclass
class CommentNode:
    comment_id: str
    nickname: str
    content: str
    like_count: int
    pictures: str
    pictures_count: str
    comment_type: str
    children: List['CommentNode']

def build_comment_tree(comments_str: str) -> List[CommentNode]:
    # 解析JSON字符串
    try:
        comments = json.loads(comments_str)
    except json.JSONDecodeError as e:
        return []

    # 创建评论ID到评论节点的映射
    comment_map: Dict[str, CommentNode] = {}
    
    # 创建父子关系映射
    children_map = defaultdict(list)
    
    # 第一遍遍历：创建所有评论节点
    for comment in comments:
        node = CommentNode(
            comment_id=comment['comment_id'],
            nickname=comment['nickname'],
            content=comment['content'],
            like_count=comment['like_count'],
            pictures=comment['pictures'],
            pictures_count=comment['pictures_count'],
            comment_type=comment['comment_type'],
            children=[]
        )
        comment_map[comment['comment_id']] = node
        
        # 记录父子关系
        target_id = comment['target_comment_id']
        if target_id:  # 如果有父评论
            children_map[target_id].append(comment['comment_id'])
    
    # 第二遍遍历：构建树结构
    root_comments = []
    for comment in comments:
        comment_id = comment['comment_id']
        node = comment_map[comment_id]
        
        # 添加子评论
        for child_id in children_map[comment_id]:
            node.children.append(comment_map[child_id])
        
        # 如果是根评论（没有父评论），加入根评论列表
        if not comment['target_comment_id']:
            root_comments.append(node)
    
    return root_comments

# 示例使用
def process_parquet_comments(df: pd.DataFrame) -> Dict[str, List[CommentNode]]:
    """
    处理DataFrame中的评论，返回每个note_id对应的评论树
    """
    result = {}
    for _, row in tqdm(df.iterrows()):
        note_id = row['note_id']
        comments = row['comments']
        if comments:  # 如果有评论
            result[note_id] = build_comment_tree(comments)
    
    return result

# 打印树结构的辅助函数
def print_comment_tree(node: CommentNode, level: int = 0):
    s = '  ' * level + f'- {node.nickname}: {node.content} ({node.like_count})'
    for child in node.children:
        s += ("\n" + print_comment_tree(child, level + 1))
    return s

def process_parquet(input_path: str, output_path: str):
    df = pq.read_table(input_path).to_pandas()
    result = process_parquet_comments(df)
    with open(output_path, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    process_parquet(
       input_path = "viewfs://hadoop-lt-cluster/home/reco_kaiworks/users/zhouyang12/data/recovlm/web_comments/p_date=20250328/part-00264-ed5ce4e5-32f5-486d-8e1c-c1fcb1b5d40a.c000",
       output_path = "results.json" 
    )