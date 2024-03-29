import unittest

from tasktemplates.template import PromptTemplate


class TestQQP(unittest.TestCase):

    EXAMPLE = {
        "question1": "How is the life of a math student? Could you describe your own experiences?",
        "question2": "Which level of prepration is enough for the exam jlpt5?",
        "label": 0
    }

    EXPECTED = {
        "input": f"qqp question1: {EXAMPLE['question1']} question2: {EXAMPLE['question2']}",
        "target": f"not_duplicate"
    }

    def test_qqp_t5(self):
        template = PromptTemplate("qqp")
        prompt_template = template.get_template(
            "T5", "qqp-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
