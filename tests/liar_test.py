import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters


class TestLIAR(unittest.TestCase):

    EXAMPLE = {
        "statement": "Says the Annies List political group supports third-trimester abortions on demand.",
        "label": 2
    }

    def test_liar_binary_factuality_mt0(self):
        template = PromptTemplate("liar")
        prompt_template = template.get_template(
            "mT0", "liar-binary-factuality-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        EXPECTED = {
            "input": f"factuality claim: {replace_whitecharacters(remove_urls(self.EXAMPLE['statement']))}",
            "target": f"True"
        }
        self.assertEqual(output, EXPECTED)

    def test_liar_multiclass_factuality_mt0(self):
        template = PromptTemplate("liar")
        prompt_template = template.get_template(
            "mT0", "liar-multiclass-factuality-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        EXPECTED = {
            "input": f"factuality claim: {replace_whitecharacters(remove_urls(self.EXAMPLE['statement']))}",
            "target": f"Mostly True"
        }
        self.assertEqual(output, EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
