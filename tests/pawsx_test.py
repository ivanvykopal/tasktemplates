import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import remove_urls, replace_whitecharacters


class TestPAWSX(unittest.TestCase):

    EXAMPLE = {
        "sentence1": "In Paris , in October 1560 , he secretly met the English ambassador , Nicolas Throckmorton , asking him for a passport to return to England through Scotland .",
        "sentence2": "In October 1560 , he secretly met with the English ambassador , Nicolas Throckmorton , in Paris , and asked him for a passport to return to Scotland through England .",
        "label": 0
    }

    def test_paws_x_mt0(self):
        template = PromptTemplate("pawsx")
        prompt_template = template.get_template(
            "mT0", "paws-x-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        EXPECTED = {
            "input": f"Sentence1: {self.EXAMPLE['sentence1']}\\nSentence2: {self.EXAMPLE['sentence2']}\\nQuestion: Do Sentence 1 and Sentence 2 express the same meaning? Yes or No?",
            "target": f"No"
        }
        self.assertEqual(output, EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
