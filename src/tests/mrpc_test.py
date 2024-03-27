import unittest

from template import PromptTemplate


class TestMRPC(unittest.TestCase):

    EXAMPLE = {
        "sentence1": "Amrozi accused his brother , whom he called \" the witness \" , of deliberately distorting his evidence .",
        "sentence2": "Referring to him as only \" the witness \" , Amrozi accused his brother of deliberately distorting his evidence .",
        "label": 1
    }

    EXPECTED = {
        "input": f"sentence1: {EXAMPLE['sentence1']} sentence2: {EXAMPLE['sentence2']}",
        "target": f"equivalent"
    }

    def test_mrpc_t5(self):
        template = PromptTemplate("mrpc")
        prompt_template = template.get_template(
            "T5", "mrpc-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
