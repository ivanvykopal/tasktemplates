import unittest

from tasktemplates.template import PromptTemplate


class TestSocialIQA(unittest.TestCase):

    EXAMPLE = {
        "question": "How would Others feel as a result?",
        "context": "Cameron decided to have a barbecue and gathered her friends together.",
        "answerA": "like attending",
        "answerB": "like staying home",
        "answerC": "a good friend to have",
        "label": 1
    }

    EXPECTED = {
        "input": f"question: {EXAMPLE['question']} context: {EXAMPLE['context']} || choice0: {EXAMPLE['answerA']} || choice1: {EXAMPLE['answerB']} || choice2: {EXAMPLE['answerC']}",
        "target": f"{str(int(EXAMPLE['label']) - 1)}"
    }

    def test_social_i_qa_t5(self):
        template = PromptTemplate("social_i_qa")
        prompt_template = template.get_template(
            "T5", "social_i_qa-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
