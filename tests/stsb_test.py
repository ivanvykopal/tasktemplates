import unittest
import numpy as np

from tasktemplates.template import PromptTemplate


class TestSTSB(unittest.TestCase):

    EXAMPLE = {
        "sentence1": "A man is playing the cello.",
        "sentence2": "A man seated is playing the cello.",
        "label": 4.25
    }

    EXPECTED = {
        "input": f"stsb sentence1: {EXAMPLE['sentence1']} sentence2: {EXAMPLE['sentence2']}",
        "target": f"{np.round((EXAMPLE['label'] * 5) / 5, decimals=1)}"
    }

    def test_stsb_t5(self):
        template = PromptTemplate("stsb")
        prompt_template = template.get_template(
            "T5", "stsb-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
