import unittest
import numpy as np

from template import PromptTemplate


class TestWNLI(unittest.TestCase):

    EXAMPLE = {
        "sentence1": "I stuck a pin through a carrot. When I pulled the pin out, it had a hole.",
        "sentence2": "The carrot had a hole.",
        "label": 1
    }
    EXPECTED = {
        "input": f"wnli sentence1: {EXAMPLE['sentence1']} sentence2: {EXAMPLE['sentence2']}",
        "target": f"entailment"
    }

    def test_wnli_t5(self):
        template = PromptTemplate("wnli")
        prompt_template = template.get_template(
            "T5", "wnli-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
