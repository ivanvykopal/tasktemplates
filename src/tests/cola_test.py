import unittest

from template import PromptTemplate


class TestCola(unittest.TestCase):

    EXAMPLE = {
        "sentence": "Our friends won't buy this analysis, let alone the next one we propose.",
        "label": 1
    }
    EXPECTED = {
        "input": f"cola sentence: {EXAMPLE['sentence']}",
        "target": f"acceptable"
    }

    def test_cola_t5(self):
        template = PromptTemplate("cola")
        prompt_template = template.get_template(
            "T5", "cola-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
