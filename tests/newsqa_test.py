import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import pad_punctuation


class TestNewsQA(unittest.TestCase):

    EXAMPLE = {
        "question": "What was the amount of children murdered?",
        "context": "NEW DELHI, India (CNN) -- A high court in northern India on Friday acquitted a wealthy businessman facing the death sentence for the killing of a teen in a case dubbed \"the house of horrors.\" Moninder Singh Pandher was sentenced to death by a lower court in February. The teen was one of 19 victims -- children and young women -- in one of the most gruesome serial killings in India in recent years. The Allahabad high court has acquitted Moninder Singh Pandher, his lawyer Sikandar B. Kochar told CNN. Pandher and his domestic employee Surinder Koli were sentenced to death in February by a lower court for the rape and murder of the 14-year-old. The high court upheld Koli's death sentence, Kochar said. The two were arrested two years ago after body parts packed in plastic bags were found near their home in Noida, a New Delhi suburb. Their home was later dubbed a \"house of horrors\" by the Indian media. Pandher was not named a main suspect by investigators initially, but was summoned as co-accused during the trial, Kochar said. Kochar said his client was in Australia when the teen was raped and killed. Pandher faces trial in the remaining 18 killings and could remain in custody, the attorney said.",
        "answer": ["19"]
    }

    EXPECTED = {
        "input": f"question: {pad_punctuation(EXAMPLE['question'])} context: {pad_punctuation(EXAMPLE['context'])}",
        "target": f"{pad_punctuation(EXAMPLE['answer'][0])}"
    }

    EXPECTED_WITHOUT_CONTEX = {
        "input": f"question: {pad_punctuation(EXAMPLE['question'])}",
        "target": f"{pad_punctuation(EXAMPLE['answer'][0])}"
    }

    def test_newsqa_t5(self):
        template = PromptTemplate("newsqa")
        prompt_template = template.get_template(
            "T5", "newsqa-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)

    def test_newsqa_t5_2(self):
        template = PromptTemplate("newsqa")
        prompt_template = template.get_template(
            "T5", "newsqa-prompt-t5-without-context")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED_WITHOUT_CONTEX)


# run test
if __name__ == '__main__':
    unittest.main()
