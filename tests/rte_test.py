import unittest

from tasktemplates.template import PromptTemplate


class TestRTE(unittest.TestCase):

    EXAMPLE = {
        "sentence1": "No Weapons of Mass Destruction Found in Iraq Yet.",
        "sentence2": "Weapons of Mass Destruction Found in Iraq.",
        "label": 1
    }

    EXPECTED = {
        "input": f"rte sentence1: {EXAMPLE['sentence1']} sentence2: {EXAMPLE['sentence2']}",
        "target": f"not_entailment"
    }

    def test_rte_t5(self):
        template = PromptTemplate("rte")
        prompt_template = template.get_template(
            "T5", "rte-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
