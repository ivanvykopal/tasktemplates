import unittest

from template import PromptTemplate


class TestSST2(unittest.TestCase):

    EXAMPLE = {
        "sentence": "hide new secretions from the parental units",
        "label": 0
    }

    EXPECTED = {
        "input": f"sentence: {EXAMPLE['sentence']}",
        "target": f"negative"
    }

    def test_sst2_t5(self):
        template = PromptTemplate("sst2")
        prompt_template = template.get_template(
            "T5", "sst2-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
