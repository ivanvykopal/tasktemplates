import unittest

from template import PromptTemplate


class TestQNLI(unittest.TestCase):

    EXAMPLE = {
        "question": "When did the third Digimon series begin?",
        "sentence": "Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese.",
        "label": 1
    }

    EXPECTED = {
        "input": f"qnli question: {EXAMPLE['question']} sentence: {EXAMPLE['sentence']}",
        "target": f"not_entailment"
    }

    def test_qnli_t5(self):
        template = PromptTemplate("qnli")
        prompt_template = template.get_template(
            "T5", "qnli-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
