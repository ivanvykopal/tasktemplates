import unittest
import numpy as np

from template import PromptTemplate


class TestWinogrande(unittest.TestCase):

    EXAMPLE = {
        "sentence": "John moved the couch from the garage to the backyard to create space. The _ is small.",
        "option1": "garage",
        "option2": "backyard",
        "answer": 1
    }
    EXPECTED = {
        "input": f"sentence: {EXAMPLE['sentence']} option0: {EXAMPLE['option1']} option1: {EXAMPLE['option2']}",
        "target": f"{str(int(EXAMPLE['answer']) - 1)}"
    }

    def test_winogrande_t5(self):
        template = PromptTemplate("winogrande")
        prompt_template = template.get_template(
            "T5", "winogrande-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
