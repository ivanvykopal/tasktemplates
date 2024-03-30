import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import pad_punctuation


class TestCSSQUAD(unittest.TestCase):

    EXAMPLE = {
        "context": "Karibská krize (též Kubánská krize) byla mezinárodní politická krize. Hrozilo, že přeroste v jaderný konflikt. Vypukla v roce 1962 v důsledku rozmístění sovětských raket středního doletu na Kubě, kterým SSSR odpověděl na umístění amerických raket v Turecku. V reakci na to vyhlásily Spojené státy americké blokádu Kuby, která měla zabránit dopravení dalších raket na toto území.",
        "question": "Ve kterém roce vypukla Karibská krize?",
        "answers": {
            "text": ["1962"]
        }
    }

    def test_cssquad_prefix_mt0(self):
        template = PromptTemplate("cssquad")
        prompt_template = template.get_template(
            "mT0", "cssquad-prefix-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        EXPECTED = {
            "input": f"question: {pad_punctuation(self.EXAMPLE['question'])} context: {pad_punctuation(self.EXAMPLE['context'])}",
            "target": f"{self.EXAMPLE['answers']['text'][0]}"
        }
        self.assertEqual(output, EXPECTED)

    def test_csquad_instruct1_mt0(self):
        template = PromptTemplate("cssquad")
        prompt_template = template.get_template(
            "mT0", "cssquad-instruct1-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        EXPECTED = {
            "input": f"Answer the question depending on the context. Context: {self.EXAMPLE['context']}; Question: {self.EXAMPLE['question']}; Answer:",
            "target": f"{self.EXAMPLE['answers']['text'][0]}"
        }
        self.assertEqual(output, EXPECTED)

    def test_csquad_instruct2_mt0(self):
        template = PromptTemplate("cssquad")
        prompt_template = template.get_template(
            "mT0", "cssquad-instruct2-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        EXPECTED = {
            "input": f"What is the answer? Context: {self.EXAMPLE['context']}; Question: {self.EXAMPLE['question']}; Answer:",
            "target": f"{self.EXAMPLE['answers']['text'][0]}"
        }
        self.assertEqual(output, EXPECTED)

    def test_csquad_instruct3_mt0(self):
        template = PromptTemplate("cssquad")
        prompt_template = template.get_template(
            "mT0", "cssquad-instruct3-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        EXPECTED = {
            "input": f"Given the following passage \"{self.EXAMPLE['context']}\", answer the following question. Note that the answer is present within the text. Question: {self.EXAMPLE['question']}",
            "target": f"{self.EXAMPLE['answers']['text'][0]}"
        }
        self.assertEqual(output, EXPECTED)

    def test_csquad_instruct4_mt0(self):
        template = PromptTemplate("cssquad")
        prompt_template = template.get_template(
            "mT0", "cssquad-instruct4-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        EXPECTED = {
            "input": f"Refer to the passage below and answer the following question: Passage: {self.EXAMPLE['context']} Question: {self.EXAMPLE['question']}",
            "target": f"{self.EXAMPLE['answers']['text'][0]}"
        }
        self.assertEqual(output, EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
