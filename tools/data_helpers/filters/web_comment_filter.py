import json
from typing import Dict
from .filter import FilterBase
from .common.comment import CommentNode, build_comment_tree

class WebCommentFilter(FilterBase):

    def __init__(self, ):
        super().__init__()

    def __call__(self, src: Dict[str, any]) -> bool:
        pass
