import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters


class TestLESA2021(unittest.TestCase):

    EXAMPLE = {
        "en": "Conservatives / Republicans , moderate Democrats . ",
        "claim": 0.0
    }

    EXPECTED = {
        "input": f"checkworthiness claim: {replace_whitecharacters(remove_urls(EXAMPLE['en']))}",
        "target": f"not_checkworthy"
    }

    def test_lesa2021_factuality_mt0(self):
        template = PromptTemplate("lesa2021")
        prompt_template = template.get_template(
            "mT0", "lesa2021-checkworthiness-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
