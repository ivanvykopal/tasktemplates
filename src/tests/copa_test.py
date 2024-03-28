import unittest

from template import PromptTemplate


class TestCopa(unittest.TestCase):

    EXAMPLE = {
        "premise": "My body cast a shadow over the grass.",
        "choice1": "The sun was rising.",
        "choice2": "The grass was cut.",
        "label": 0
    }
    EXPECTED = {
        "input": f"copa premise: {EXAMPLE['premise']} choice1: {EXAMPLE['choice1']} choice2: {EXAMPLE['choice2']}",
        "target": f"choice1"
    }

    def test_copa_t5(self):
        template = PromptTemplate("copa")
        prompt_template = template.get_template(
            "T5", "copa-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
