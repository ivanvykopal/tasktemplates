import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters


class TestClaimBuster(unittest.TestCase):

    EXAMPLE = {
        "text": "You know, I saw a movie - \"Crocodile Dundee.\"",
        "label": 0
    }

    EXPECTED = {
        "input": f"checkworthiness claim: {replace_whitecharacters(remove_urls(EXAMPLE['text']))}",
        "target": f"not_checkworthy"
    }

    def test_claimbuster_mt0(self):
        template = PromptTemplate("claimbuster")
        prompt_template = template.get_template(
            "mT0", "claimbuster-checkworthiness-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
