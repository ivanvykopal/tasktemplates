import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters


class TestDemagog(unittest.TestCase):

    EXAMPLE = {
        "claim": "My (Most-Híd, pozn.) sme robili tiež kampaň priamo v regiónoch, ako hovorí pán Beblavý.",
        "label": "Pravda"
    }

    EXPECTED = {
        "input": f"factuality claim: {replace_whitecharacters(remove_urls(EXAMPLE['claim']))}",
        "target": f"True"
    }

    def test_demagog_factuality_mt0(self):
        template = PromptTemplate("demagog")
        prompt_template = template.get_template(
            "mT0", "demagog-factuality-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
