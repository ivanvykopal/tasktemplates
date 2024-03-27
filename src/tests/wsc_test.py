import unittest
import numpy as np

from template import PromptTemplate


class TestWSC(unittest.TestCase):

    EXAMPLE = {
        'text': 'This is a test sentence .',
        'span1_text': 'test',
        'span1_index': 3,
        'span2_text': 'This',
        'span2_index': 0,
        'label': 0
    }
    EXPECTED = {
        "input": f"wsc text: # This # is a * test * sentence .",
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
