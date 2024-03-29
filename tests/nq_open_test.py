import unittest

from tasktemplates.template import PromptTemplate


class TestNQOpen(unittest.TestCase):

    EXAMPLE = {
        "question": "where did they film hot tub time machine",
        "answer": ["Fernie Alpine Resort"]
    }

    EXPECTED = {
        "input": f"nq question: {EXAMPLE['question']}",
        "target": f"{EXAMPLE['answer'][0]}"
    }

    def test_nq_open_t5(self):
        template = PromptTemplate("nq_open")
        prompt_template = template.get_template(
            "T5", "nq_open-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
