import json
from typing import Dict, List
from .converter import (
    ConverterBase,
    iter_messages,
    iter_segments,
)
import selectolax

class CleanHtmlConverter(ConverterBase):

    def __call__(self, src: Dict[str, any]) -> Dict[str, any]:
        src['messages'] = iter_messages(src['messages'], self.clean_html)
        src['segments'] = iter_segments(src['segments'], self.clean_html)
        return src
    
    def clean_html(self, text):
        text = text.replace('<li>', '\n*')
        text = text.replace('</li>', '')
        text = text.replace('<ol>', '\n*')
        text = text.replace('</ol>', '')
        parser = selectolax.parser.HTMLParser(text)
        return parser.text()