import unittest
import numpy as np

from template import PromptTemplate


class TestXSUM(unittest.TestCase):

    EXAMPLE = {
        "document": "Mark told Pete many lies about himself, which Pete included in his book. He should have been more skeptical.",
        "target": "This is a summary of the document."
    }
    EXPECTED = {
        "input": f"summarize: {EXAMPLE['document']}",
        "target": f"{EXAMPLE['target']}"
    }

    def test_xsum_t5(self):
        template = PromptTemplate("xsum")
        prompt_template = template.get_template(
            "T5", "xsum-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
