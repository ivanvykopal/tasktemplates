import unittest

from template import PromptTemplate


class TestMNLI(unittest.TestCase):

    EXAMPLE = {
        "premise": "Conceptually cream skimming has two basic dimensions - product and geography.",
        "hypothesis": "Product and geography are what make cream skimming work.",
        "label": 1
    }

    EXPECTED = {
        "input": f"mnli premise: {EXAMPLE['premise']} hypothesis: {EXAMPLE['hypothesis']}",
        "target": f"neutral"
    }

    def test_mnli_t5(self):
        template = PromptTemplate("mnli")
        prompt_template = template.get_template(
            "T5", "mnli-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
