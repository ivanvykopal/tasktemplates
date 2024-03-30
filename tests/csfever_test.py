import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters


class TestCSFEVER(unittest.TestCase):

    EXAMPLE = {
        "claim": "Nikolaj Coster-Waldau spolupracoval se společností Fox Broadcasting Company.",
        "evidence": "Fox Broadcasting Company (často zkracováno na Fox a stylizováno jako FOX) je americká anglicky vysílající komerční televizní síť, která je vlastněna dceřinou společností Fox Entertainment Group společnosti 21st Century Fox. Nikolaj Coster-Waldau. Poté hrál detektiva Johna Amsterdama v krátkometrážním seriálu televize Fox New Amsterdam (2008) a také se objevil jako Frank Pike v televizním filmu televize Fox Virtuality z roku 2009, který byl původně zamýšlen jako pilotní film. Do povědomí širokého publika se dostal díky své současné roli sera Jaimeho Lannistera v seriálu HBO Hra o trůny.",
        "label": 2
    }

    EXPECTED = {
        "input": f"factuality claim: {replace_whitecharacters(remove_urls(EXAMPLE['claim']))} evidence: {replace_whitecharacters(remove_urls(EXAMPLE['evidence']))}",
        "target": f"SUPPORTS"
    }

    def test_csfever_factuality_mt0(self):
        template = PromptTemplate("csfever")
        prompt_template = template.get_template(
            "mT0", "csfever-factuality-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
