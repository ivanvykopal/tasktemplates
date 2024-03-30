import unittest
import numpy as np

from tasktemplates.template import PromptTemplate


class TestXNLI(unittest.TestCase):

    EXAMPLE = {
        "premise": "Conceptually cream skimming has two basic dimensions - product and geography .",
        "hypothesis": "Product and geography are what make cream skimming work .",
        "label": 1
    }
    EXPECTED = {
        "input": f"{EXAMPLE['premise']}\\nQuestion: {EXAMPLE['hypothesis']} True, False or Neither?",
        "target": f"Neither"
    }

    def test_xnli_mt0(self):
        template = PromptTemplate("xnli")
        prompt_template = template.get_template(
            "mT0", "xnli-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
