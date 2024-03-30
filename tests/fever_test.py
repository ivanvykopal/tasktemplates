import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters


class TestFEVER(unittest.TestCase):

    EXAMPLE = {
        "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
        "label": "SUPPORTS"
    }

    EXPECTED = {
        "input": f"factuality claim: {replace_whitecharacters(remove_urls(EXAMPLE['claim']))}",
        "target": f"True"
    }

    def test_fever_factuality_mt0(self):
        template = PromptTemplate("fever")
        prompt_template = template.get_template(
            "mT0", "fever-factuality-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
