import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters, pad_punctuation


class TestXFact(unittest.TestCase):

    EXAMPLE = {
        "evidence_1": "Oct 30, 2018 ... Resources ... From the start, Gillum's underdog campaign for governor of Florida was ... To those who have known and watched him closely, his gift of ... \"The last five or six candidates [Democrats] put up for governor, they've been like .... family's cycle of intergenerational poverty, Gillum became the first in ...",
        "evidence_2": "Explore legal resources, campaign finance data, help for candidates and ... Summary of 21-Month Campaign Activity of the 2017-2018 Election Cycle ... Learn about how much contributors can give to different types of committees ... By law, no more than three can represent the same political party. ... 1050 First Street, NE",
        "evidence_3": "In American politics, the term swing state refers to any state that could reasonably be won by either the Democratic or Republican presidential candidate. These states are usually targeted by both major-party campaigns, especially ... A campaign strategy centered on them, however, would not have been ..... Categories:.",
        "evidence_4": "Oct 31, 2018 ... President Trump campaigns for Republican candidates at a rally in Estero, ... WE' VE SHOWN OUR VIEWERS A MAP OF WHERE THESE .... SO CALIFORNIA HAS A REALLY INTERESTING KIND OF ... WHERE DEMOCRATS HAD HOPED TO COMPETE THIS CYCLE, OR ... ACTUALLY, THAT'S NOT 49.",
        "evidence_5": "Jul 11, 2018 ... Preregistration: In Florida, preregistration laws have been found to ... Eliminating early voting has also been found to decrease turnout in communities of color. ... Through them, the 92 million eligible voters who did not vote in the .... Each election cycle, barriers to the voter registration process—including a ...",
        "claim": "No Democratic campaign for (Fla.) governor has ever had these kinds of resources this early on in an election cycle.",
        "label": "true",
    }

    def test_xfact_mt0(self):
        template = PromptTemplate("xfact")
        prompt_template = template.get_template(
            "mT0", "xfact-factuality-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)

        EXPECTED = {
            "input": f"factuality claim: {remove_urls(replace_whitecharacters(self.EXAMPLE['claim']))}",
            "target": f"True"
        }
        self.assertEqual(output, EXPECTED)

    def test_xfact_evidence_mt0(self):
        template = PromptTemplate("xfact")
        prompt_template = template.get_template(
            "mT0", "xfact-factuality-evidence-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)

        EXPECTED = {
            "input": f"factuality claim: {pad_punctuation(remove_urls(replace_whitecharacters(self.EXAMPLE['claim'])))} evidence1: {pad_punctuation(remove_urls(replace_whitecharacters(self.EXAMPLE['evidence_1'])))} evidence2: {pad_punctuation(remove_urls(replace_whitecharacters(self.EXAMPLE['evidence_2'])))} evidence3: {pad_punctuation(remove_urls(replace_whitecharacters(self.EXAMPLE['evidence_3'])))} evidence4: {pad_punctuation(remove_urls(replace_whitecharacters(self.EXAMPLE['evidence_4'])))} evidence5: {pad_punctuation(remove_urls(replace_whitecharacters(self.EXAMPLE['evidence_5'])))}",
            "target": f"True"
        }
        self.assertEqual(output, EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
