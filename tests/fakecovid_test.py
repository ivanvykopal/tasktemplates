import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters


class TestFakeCovid(unittest.TestCase):

    EXAMPLE = {
        "source_title": "Detector a video falso que dice que el Covid es una ""bacteria amplificada"" relacionada con 5G | La Silla Vacía",
        "class": 0
    }

    EXPECTED = {
        "input": f"factuality claim: {replace_whitecharacters(remove_urls(EXAMPLE['source_title']))}",
        "target": f"False"
    }

    def test_fakecovid_factuality_mt0(self):
        template = PromptTemplate("fakecovid")
        prompt_template = template.get_template(
            "mT0", "fakecovid-factuality-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
