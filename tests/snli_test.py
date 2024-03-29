import unittest

from tasktemplates.template import PromptTemplate


class TestSNLI(unittest.TestCase):

    EXAMPLE = {
        "premise": "A person on a horse jumps over a broken down airplane.",
        "hypothesis": "A person is training his horse for a competition.",
        "label": 1
    }

    EXPECTED = {
        "input": f"premise: {EXAMPLE['premise']} hypothesis: {EXAMPLE['hypothesis']}",
        "target": f"neutral"
    }

    def test_snli_t5(self):
        template = PromptTemplate("snli")
        prompt_template = template.get_template(
            "T5", "snli-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
