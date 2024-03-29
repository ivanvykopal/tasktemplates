import unittest

from tasktemplates.template import PromptTemplate


class TestPiqa(unittest.TestCase):

    EXAMPLE = {
        "goal": "When boiling butter, when it's ready, you can",
        "sol1": "Pour it onto a plate",
        "sol2": "Pour it into a jar",
        "label": 1
    }

    EXPECTED = {
        "input": f"question: {EXAMPLE['goal']} choice1: {EXAMPLE['sol1']} choice2: {EXAMPLE['sol2']}",
        "target": f"{str(EXAMPLE['label'])}"
    }

    def test_piqa_t5(self):
        template = PromptTemplate("piqa")
        prompt_template = template.get_template(
            "T5", "piqa-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
