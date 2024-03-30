import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters


class TestHover(unittest.TestCase):

    EXAMPLE = {
        "claim": "Skagen Painter Peder Severin Krøyer favored naturalism along with Theodor Esbern Philipsen and the artist Ossian Elgström studied with in the early 1900s.",
        "label": 1
    }

    EXPECTED = {
        "input": f"factuality claim: {replace_whitecharacters(remove_urls(EXAMPLE['claim']))}",
        "target": f"True"
    }

    def test_hover_factuality_mt0(self):
        template = PromptTemplate("hover")
        prompt_template = template.get_template(
            "mT0", "hover-factuality-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
