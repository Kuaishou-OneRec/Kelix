import re
import json
from typing import Dict, List, Optional
from .converter import (
    ConverterBase,
    iter_messages,
    iter_segments,
)

class CleanLinksConverter(ConverterBase):

    def __init__(self, pattern: Optional[str] = None, repl: str = ''):
        if pattern is None:
            self.pattern = r'(?i)\b('
            self.pattern += r'(?:[a-z][\w-]+:(?:\/{1,3}|'
            self.pattern += r'[a-z0-9%])|www\d{0,3}[.]|'
            self.pattern += r'[a-z0-9.\-]+[.][a-z]{2,4}\/)'
            self.pattern += r'(?:[^\s()<>]+|\(([^\s()<>]+|'
            self.pattern += r'(\([^\s()<>]+\)))*\))'
            self.pattern += r'+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|'
            self.pattern += r'[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])'
            self.pattern += r')'
        else:
            self.pattern = pattern
        
        self.repl = repl


    def __call__(self, src: Dict[str, any]) -> Dict[str, any]:
        src['messages'] = iter_messages(src['messages'], self.clean_links)
        src['segments'] = iter_segments(src['segments'], self.clean_links)
        return src
    
    def clean_links(self, text):
        if not re.search(self.pattern, text, flags=re.DOTALL):
            return text
        rst = re.sub(pattern=self.pattern, repl=self.repl, string=text, flags=re.DOTALL)
        return rst
