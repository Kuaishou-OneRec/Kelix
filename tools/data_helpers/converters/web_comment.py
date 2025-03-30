import os
import os.path as osp
import json
import argparse
import subprocess
import tempfile
import numpy as np
import random
from typing import Dict, List, Tuple, Set, Optional, List
from .converter import ConverterBase
from recovlm.utils.blobstore_client import BlobStoreClient
import glob
import uuid
import cv2
import base64
from PIL import Image
import traceback
import re
import random
import pandas as pd

from collections import defaultdict
from dataclasses import dataclass

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

    def get_total_replies(self) -> int:
        total = len(self.children)
        for child in self.children:
            total += child.get_total_replies()
        return total

    def get_total_likes(self) -> int:
        total = self.like_count
        for child in self.children:
            total += child.get_total_likes()
        return total

def build_comment_tree(comments_str: str) -> List[CommentNode]:
    try:
        comments = json.loads(comments_str)
    except json.JSONDecodeError:
        return []

    comment_map: Dict[str, CommentNode] = {}

    children_map = defaultdict(list)

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

        target_id = comment['target_comment_id']
        if target_id:
            children_map[target_id].append(comment['comment_id'])

    root_comments = []
    for comment in comments:
        comment_id = comment['comment_id']
        node = comment_map[comment_id]

        for child_id in children_map[comment_id]:
            node.children.append(comment_map[child_id])

        if not comment['target_comment_id']:
            root_comments.append(node)
    
    return root_comments

def clean_comment_content(content: str) -> str:
    """清理评论内容
    
    1. 去除[搜索高亮]
    2. 去除首尾空白
    """
    if not content:
        return ""
    
    # 去除[搜索高亮]
    content = content.replace("[搜索高亮]", "")
    # 去除首尾空白
    return content.strip()

def is_valid_comment(node: CommentNode) -> bool:
    """检查评论是否符合筛选条件
    
    筛选条件：
    1. 不包含图片
    2. 不包含@xxx（评论中任何位置都不能有@）
    3. 内容不为空
    """
    # 检查是否包含图片
    if node.pictures or node.pictures_count:
        return False
    
    # 清理评论内容
    cleaned_content = clean_comment_content(node.content)
    
    # 检查内容是否为空
    if not cleaned_content:
        return False
    
    # 检查是否包含@（任何位置）
    if '@' in cleaned_content:
        return False
    
    return True

def filter_top_comments(comment_trees: List[CommentNode], min_likes: int = 100, top_k: int = 10) -> List[CommentNode]:
    """筛选点赞数超过阈值的top-k一级评论
    
    Args:
        comment_trees: 评论树列表
        min_likes: 最小点赞数阈值
        top_k: 保留的评论数量
    
    Returns:
        筛选后的一级评论列表
    """
    # 筛选符合条件的评论
    filtered_comments = [
        comment for comment in comment_trees
        if (comment.like_count >= min_likes and is_valid_comment(comment))
    ]
    
    # 按点赞数排序并返回top-k
    sorted_comments = sorted(filtered_comments, key=lambda x: x.like_count, reverse=True)
    return sorted_comments[:top_k]

def get_tree_stats(comment_tree: List[CommentNode]) -> Tuple[int, int]:
    """获取评论树的总点赞数和总回复数"""
    total_likes = 0
    total_replies = 0
    for root_comment in comment_tree:
        total_likes += root_comment.get_total_likes()
        total_replies += root_comment.get_total_replies()
    return total_likes, total_replies

class WebCommentConverter(ConverterBase):

    def __init__(
        self,
        prompts,
        source,
        **kwargs
    ):
        self.prompts = prompts
        self.source = source
        self.min_likes = kwargs.get("min_likes", 100)
        self.top_k = kwargs.get("top_k", 10)
            
    def __call__(self, src: Dict[str, any]) -> Optional[List[Dict[str, any]]]:
        note_id = src['note_id']
        comments_str = src['comments']

        if not comments_str:
            return []

        # 构建评论树
        try:
            comment_trees = build_comment_tree(comments_str)
            if not comment_trees:  # JSON解析失败会返回空列表
                return []
        except Exception as e:
            return []
        
        # 计算note的总点赞数
        total_likes, _ = get_tree_stats(comment_trees)
        
        # 筛选条件：note总点赞数>=100
        if total_likes < self.min_likes:
            return []
            
        # 筛选top评论并按点赞量排序
        top_comments = filter_top_comments(
            comment_trees, min_likes=self.min_likes, top_k=self.top_k)
        
        # 如果没有符合条件的评论，跳过该note
        if not top_comments:
            return []

        prompt = np.random.choice(self.prompts)
        
        doc = f"标题：{src['note_title']}\n内容：{src['note_desc']}\n"
        results = []
        for comment in top_comments:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": doc
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": comment.content
                        }
                    ]
                }
            ]
            meta = {
                "source": self.source,
                "images": json.dumps({}),
                "videos": json.dumps({}),
                "segments": None,
                "metadata": None,
                "messages": json.dumps(messages),
                "uuid": str(uuid.uuid1())
            }
            results.append(meta)
        return results
