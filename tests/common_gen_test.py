import unittest

from tasktemplates.template import PromptTemplate


class TestCommonGen(unittest.TestCase):

    EXAMPLE = {
        "concepts": ["mountain", "ski", "skier"],
        "target": "Skier skis down the mountain"
    }
    EXPECTED = {
        "input": f"generate: {' '.join(EXAMPLE['concepts'])}",
        "target": f"{EXAMPLE['target']}"
    }

    def test_common_gen_t5(self):
        template = PromptTemplate("common_gen")
        prompt_template = template.get_template(
            "T5", "common_gen-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
