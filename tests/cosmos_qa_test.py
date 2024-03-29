import unittest

from tasktemplates.template import PromptTemplate


class TestCosmosQA(unittest.TestCase):

    EXAMPLE = {
        "context": "Good Old War and person L : I saw both of these bands Wednesday night , and they both blew me away . seriously . Good Old War is acoustic and makes me smile . I really can not help but be happy when I listen to them ; I think it 's the fact that they seemed so happy themselves when they played .",
        "question": "In the future , will this person go to see other bands play ?",
        "answer0": "None of the above choices .",
        "answer1": "This person likes music and likes to see the show , they will see other bands play .",
        "answer2": "This person only likes Good Old War and Person L , no other bands .",
        "answer3": "Other Bands is not on tour and this person can not see them .",
        "label": 1
    }
    EXPECTED = {
        "input": f"question: {EXAMPLE['question']} context: {EXAMPLE['context']} choice0: {EXAMPLE['answer0']} choice1: {EXAMPLE['answer1']} choice2: {EXAMPLE['answer2']} choice3: {EXAMPLE['answer3']}",
        "target": f"{str(EXAMPLE['label'])}"
    }

    def test_cosmos_qa_t5(self):
        template = PromptTemplate("cosmos_qa")
        prompt_template = template.get_template(
            "T5", "cosmos_qa-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
