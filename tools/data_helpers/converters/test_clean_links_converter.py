import json
import unittest

from .clean_links_converter import CleanLinksConverter

class TestCleanLinksConverter(unittest.TestCase):

    def test_case(self):
        case0 = {
            "segments": json.dumps([
                {"type": "text", "text": "hello world"}
            ]),
            "messages": json.dumps([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "https://xxxx"}
                    ]
                }
            ])
        }

        cvt = CleanLinksConverter()
        rst = cvt(case0)
        tgt = {'segments': '[{"type": "text", "text": "hello world"}]', 'messages': '[{"role": "user", "content": [{"type": "text", "text": ""}]}]'}
        self.assertEqual(json.dumps(rst), json.dumps(tgt))
    
if __name__ == '__main__':
    unittest.main()