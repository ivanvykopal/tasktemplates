import unittest
import numpy as np

from template import PromptTemplate


class TestWSC(unittest.TestCase):

    EXAMPLE = {
        "text": "Mark told Pete many lies about himself, which Pete included in his book. He should have been more skeptical.",
        "label": 0
    }
    EXPECTED = {
        "input": f"wsc text: {EXAMPLE['text']}",
        "target": f"False"
    }

    def test_wsc_t5(self):
        template = PromptTemplate("wsc")
        prompt_template = template.get_template(
            "T5", "wsc-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
