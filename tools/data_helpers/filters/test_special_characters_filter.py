import json
import unittest

from .special_characters_filter import SpecialCharactersFilter


class TestSpecialCharactersFilter(unittest.TestCase):

    def test_case(self):
        case0 = {
            "segments": json.dumps([
                {"type": "text", "text": "@!@# $@#@!!! ab"}
            ]),
            "messages": "null"
        }
        case1 = {
            "segments": "null",
            "messages": json.dumps([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "this is a test case."},
                    ],
                }
            ]),
        }
        f = SpecialCharactersFilter(min_ratio=0.0, max_ratio=0.25)

        self.assertFalse(f(case0))
        self.assertTrue(f(case1))

if __name__ == '__main__':
    unittest.main()