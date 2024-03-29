import unittest

from tasktemplates.template import PromptTemplate


class TestGigaword(unittest.TestCase):

    EXAMPLE = {
        "document": "australia 's current account deficit shrunk by a record #.## billion dollars -lrb- #.## billion us -rrb- in the june quarter due to soaring commodity prices , figures released monday showed .",
        "summary": "australian current account deficit narrows sharply"
    }

    EXPECTED = {
        "input": f"summarize: {EXAMPLE['document']}",
        "target": f"{EXAMPLE['summary']}"
    }

    def test_gigaword_t5(self):
        template = PromptTemplate("gigaword")
        prompt_template = template.get_template(
            "T5", "gigaword-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
