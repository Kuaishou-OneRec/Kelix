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

    def get_total_replies(self) -> int:
        """统计当前评论节点下的总回复数（包括所有子孙评论）"""
        total = len(self.children)  # 直接子评论数量
        for child in self.children:
            total += child.get_total_replies()  # 递归统计子评论的回复数
        return total

    def get_total_likes(self) -> int:
        """统计当前评论节点下的总点赞数（包括所有子孙评论）"""
        total = self.like_count  # 当前评论的点赞数
        for child in self.children:
            total += child.get_total_likes()  # 递归统计子评论的点赞数
        return total

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
    s = '  ' * level + f'- {node.nickname}: {node.content}'
    s += f' (点赞: {node.like_count}, 子评论数: {len(node.children)})'
    for child in node.children:
        s += ("\n" + print_comment_tree(child, level + 1))
    return s

def get_tree_stats(comment_tree: List[CommentNode]) -> tuple[int, int]:
    """获取评论树的总点赞数和总回复数"""
    total_likes = 0
    total_replies = 0
    for root_comment in comment_tree:
        total_likes += root_comment.get_total_likes()
        total_replies += root_comment.get_total_replies()
    return total_likes, total_replies

def print_sorted_comment_trees(note_id: str, comment_trees: List[CommentNode]):
    """打印单个note下的评论树，按根节点点赞量排序
    
    Args:
        note_id: 笔记ID
        comment_trees: 评论树列表
    """
    # 按根节点点赞量排序
    sorted_trees = sorted(comment_trees, key=lambda x: x.like_count, reverse=True)
    
    print(f"\n=== Note ID: {note_id} ===")
    total_likes, total_replies = get_tree_stats(comment_trees)
    print(f"笔记总点赞数: {total_likes}, 总回复数: {total_replies}")
    print(f"评论数量: {len(comment_trees)}")
    print("\n评论树结构(按点赞量排序):")
    
    for i, root_comment in enumerate(sorted_trees, 1):
        print(f"\n[评论 {i}] 点赞数: {root_comment.like_count}, 总回复数: {root_comment.get_total_replies()}")
        print(print_comment_tree(root_comment))
    print("-" * 50)

def print_top_comment_trees(note_comments: Dict[str, List[CommentNode]], top_k: int = 10):
    """打印点赞量最高的评论树
    
    Args:
        note_comments: note_id到评论树的映射
        top_k: 打印前k个点赞最多的评论树
    """
    # 计算每个评论树的统计信息
    tree_stats = []
    for note_id, comment_tree in note_comments.items():
        total_likes, total_replies = get_tree_stats(comment_tree)
        tree_stats.append((note_id, comment_tree, total_likes, total_replies))
    
    # 按总点赞数排序
    tree_stats.sort(key=lambda x: x[2], reverse=True)
    
    # 打印top-k的评论树
    print(f"\n=== Top {top_k} Most Liked Notes ===")
    for i, (note_id, comment_tree, total_likes, total_replies) in enumerate(tree_stats[:top_k], 1):
        print(f"\n{i}. Note ID: {note_id}")
        print(f"总点赞数: {total_likes}, 总回复数: {total_replies}")
        # 使用新的打印函数
        print_sorted_comment_trees(note_id, comment_tree)

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


def comment_node_to_dict(node: CommentNode) -> dict:
    """将CommentNode对象转换为可序列化的字典"""
    return {
        'comment_id': node.comment_id,
        'nickname': node.nickname,
        'content': clean_comment_content(node.content),  # 使用清理后的内容
        'like_count': node.like_count,
        'comment_type': node.comment_type,
        'total_replies': node.get_total_replies(),
        'total_likes': node.get_total_likes(),
        # 'children_ids': [
        #     child.comment_id for child in node.children 
        #     if is_valid_comment(child)  # 只保留有效的子评论ID
        # ]
    }

def process_parquet(input_path: str, output_path: str):
    """处理parquet文件，筛选并保存符合条件的数据
    
    筛选条件：
    1. 一级评论点赞数 >= 100
    2. 每个note保留点赞数最高的前10个一级评论
    3. note的总点赞数 >= 100
    """
    df = pq.read_table(input_path).to_pandas()
    
    # 统计信息
    stats = {
        'total_notes': len(df),
        'empty_comments': 0,  # 没有comments字段的note
        'parse_failed': 0,    # JSON解析失败的note
        'low_likes': 0,       # 点赞数不够的note
        'no_valid_comments': 0,  # 没有有效评论的note
        'success': 0          # 成功处理的note
    }
    
    filtered_results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        note_id = row['note_id']
        comments_str = row['comments']
        
        if not comments_str:  # 跳过没有评论的note
            stats['empty_comments'] += 1
            continue
            
        # 构建评论树
        try:
            comment_trees = build_comment_tree(comments_str)
            if not comment_trees:  # JSON解析失败会返回空列表
                stats['parse_failed'] += 1
                continue
        except Exception as e:
            stats['parse_failed'] += 1
            continue
        
        # 计算note的总点赞数
        total_likes, _ = get_tree_stats(comment_trees)
        
        # 筛选条件：note总点赞数>=100
        if total_likes < 100:
            stats['low_likes'] += 1
            continue
            
        # 筛选top评论并按点赞量排序
        top_comments = filter_top_comments(comment_trees, min_likes=100, top_k=10)
        
        # 如果没有符合条件的评论，跳过该note
        if not top_comments:
            stats['no_valid_comments'] += 1
            continue
            
        # 保存note信息和筛选后的评论
        filtered_note = {
            'note_id': note_id,
            'total_likes': total_likes,
            'note_data': {
                col: row[col] for col in df.columns 
                if col not in ['note_id', 'comments']
            },
            'top_comments': [comment_node_to_dict(comment) for comment in top_comments]
        }
        
        filtered_results.append(filtered_note)
        stats['success'] += 1
    
    # 按note的总点赞量排序
    filtered_results.sort(key=lambda x: x['total_likes'], reverse=True)
    
    # 打印统计信息
    print("\n=== 处理统计 ===")
    print(f"总note数量: {stats['total_notes']}")
    print(f"成功处理: {stats['success']}")
    print("\n被过滤原因统计:")
    print(f"- 空评论数据: {stats['empty_comments']}")
    print(f"- 解析失败: {stats['parse_failed']}")
    print(f"- 点赞数不足: {stats['low_likes']}")
    print(f"- 无有效评论: {stats['no_valid_comments']}")
    
    # 计算成功率
    success_rate = (stats['success'] / stats['total_notes']) * 100
    print(f"\n处理成功率: {success_rate:.2f}%")
    
    # 保存结果
    print(f"\n=== 结果统计 ===")
    print(f"保存的note数量: {len(filtered_results)}")
    if filtered_results:
        print(f"第一条note的总点赞数: {filtered_results[0]['total_likes']}")
        print(f"最后一条note的总点赞数: {filtered_results[-1]['total_likes']}")
    
    # 将统计信息也保存到结果文件中
    output_data = {
        'stats': stats,
        'data': filtered_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    process_parquet(
       input_path = "viewfs://hadoop-lt-cluster/home/reco_kaiworks/users/zhouyang12/data/recovlm/web_comments/p_date=20250328/part-00264-ed5ce4e5-32f5-486d-8e1c-c1fcb1b5d40a.c000",
       output_path = "filtered_results.json" 
    )