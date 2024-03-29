import unittest

from tasktemplates.template import PromptTemplate


class TestC4(unittest.TestCase):

    EXAMPLE = {
        "text": "Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class for everyone who wants to get better with their culinary skills. He will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information. The cost to be in the class is $35 per person, and for spectators it is free. Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared.",
    }

    EXPECTED = {
        "input": f"",
        "target": f"{EXAMPLE['text']}"
    }

    def test_c4_t5(self):
        template = PromptTemplate("c4")
        prompt_template = template.get_template("T5", "c4-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
