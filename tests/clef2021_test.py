import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters


class TestCLEF2021(unittest.TestCase):

    EXAMPLE = {
        "tweet_text": "The Senate just passed COVID relief.   ✔️ $1,400 relief checks. ✔️ Funding for vaccines. ✔️ Money to reopen schools. ✔️ Food, unemployment, and rental assistance. ✔️ Cutting child poverty in half. ✔️ Help for small businesses.   We must end this pandemic. And help is on the way.",
        "check_worthiness": 0
    }

    EXPECTED = {
        "input": f"checkworthiness claim: {replace_whitecharacters(remove_urls(EXAMPLE['tweet_text']))}",
        "target": f"not_checkworthy"
    }

    def test_clef2021_checkworthiness_mt0(self):
        template = PromptTemplate("clef2021")
        prompt_template = template.get_template(
            "mT0", "clef2021-checkworthiness-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
