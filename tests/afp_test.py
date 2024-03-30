import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters


class TestAFP(unittest.TestCase):

    EXAMPLE = {
        "claim": "Former Madhya Pradesh chief minister crying after losing position.",
        "label": 0
    }

    EXPECTED = {
        "input": f"factuality claim: {replace_whitecharacters(remove_urls(EXAMPLE['claim']))}",
        "target": f"False"
    }

    def test_afp_mt0(self):
        template = PromptTemplate("afp")
        prompt_template = template.get_template(
            "mT0", "afp-factuality-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
