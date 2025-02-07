import json
import unittest

from .clean_html_converter import CleanHtmlConverter

class TestCleanHtmlConverter(unittest.TestCase):

    def test_case(self):
        case0 = {
            "segments": json.dumps([
                {"type": "text", "text": "<p1>hello world</p1>"}
            ]),
            "messages": json.dumps([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<image>https://xxxx</image>"}
                    ]
                }
            ])
        }

        cvt = CleanHtmlConverter()        

        rst = cvt(case0)
        tgt = {'segments': '[{"type": "text", "text": "hello world"}]', 'messages': '[{"role": "user", "content": [{"type": "text", "text": "https://xxxx"}]}]'}
        self.assertEqual(json.dumps(rst), json.dumps(tgt))

if __name__ == '__main__':
    unittest.main()