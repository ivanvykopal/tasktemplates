import unittest
import numpy as np

from template import PromptTemplate


class TestWIC(unittest.TestCase):

    EXAMPLE = {
        "sentence1": "Do you want to come over to my place later?",
        "sentence2": "A political system with no place for the less prominent groups.",
        "word": "place",
        "label": 0
    }

    EXPECTED = {
        "input": f"sentence1: {EXAMPLE['sentence1']} sentence2: {EXAMPLE['sentence2']} word: {EXAMPLE['word']}",
        "target": f"False"
    }

    def test_wic_t5(self):
        template = PromptTemplate("wic")
        prompt_template = template.get_template(
            "T5", "wic-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
