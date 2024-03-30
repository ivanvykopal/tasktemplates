import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters


class TestCTKFACTS(unittest.TestCase):

    EXAMPLE = {
        "claim": "Rekordní teploty se od roku 1775 měří v Praze.",
        "evidence": "PRAHA 18. června (ČTK) - Rekordní teploty 19. června (od roku 1775 měřené v pražském Klementinu) byly následující: nejvyšší teplota 31,2 z roku 1917 a 1934, nejnižší teplota 7,3 z roku 1985\. Dlouhodobý průměrný normál: 17,9 stupně Celsia.",
        "label": 2
    }

    EXPECTED = {
        "input": f"factuality claim: {replace_whitecharacters(remove_urls(EXAMPLE['claim']))} evidence: {replace_whitecharacters(remove_urls(EXAMPLE['evidence']))}",
        "target": f"SUPPORTS"
    }

    def test_ctkfacts_factuality_mt0(self):
        template = PromptTemplate("ctkfacts")
        prompt_template = template.get_template(
            "mT0", "ctkfacts-factuality-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
