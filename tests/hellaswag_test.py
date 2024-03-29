import unittest

from tasktemplates.template import PromptTemplate


class TestHellaSwag(unittest.TestCase):

    EXAMPLE = {
        "ctx": "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then",
        "endings": [", the man adds wax to the windshield and cuts it.", ", a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.", ", the man puts on a christmas coat, knitted with netting.", ", the man continues removing the snow on his car."],
        "label": 3
    }

    EXPECTED = {
        "input": f"context: {EXAMPLE['ctx']} ending0: {EXAMPLE['endings'][0]} ending1: {EXAMPLE['endings'][1]} ending2: {EXAMPLE['endings'][2]} ending3: {EXAMPLE['endings'][3]}",
        "target": f"{str(EXAMPLE['label'])}"
    }

    def test_hellaswag_t5(self):
        template = PromptTemplate("hellaswag")
        prompt_template = template.get_template(
            "T5", "hellaswag-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
