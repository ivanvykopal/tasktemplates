import unittest

from template import PromptTemplate


class TestCB(unittest.TestCase):

    EXAMPLE = {
        "premise": "It was a complex language. Not written down but handed down. One might say it was peeled down.",
        "hypothesis": "the language was peeled down",
        "label": 0
    }

    EXPECTED = {
        "input": f"cb premise: {EXAMPLE['premise']} hypothesis: {EXAMPLE['hypothesis']}",
        "target": f"entailment"
    }

    def test_cb_t5(self):
        template = PromptTemplate("cb")
        prompt_template = template.get_template("T5", "cb-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
