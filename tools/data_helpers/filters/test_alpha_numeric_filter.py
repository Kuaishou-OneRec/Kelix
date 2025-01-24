import json
import unittest
from .alpha_numeric_filter import AlphaNumericFilter

class TestAlphaNumericFilter(unittest.TestCase):

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
        alpha_numeric_filter = AlphaNumericFilter(min_ratio=0.2, max_ratio=1.0)

        self.assertFalse(alpha_numeric_filter(case0))
        self.assertTrue(alpha_numeric_filter(case1))


if __name__ == '__main__':
    unittest.main()