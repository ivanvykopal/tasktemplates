import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import pad_punctuation


class TestSearchQA(unittest.TestCase):

    EXAMPLE = {
        "question": "For the last 8 years of his life, Galileo was under house arrest for espousing this man's theory",
        "answer": "Copernicus",
    }

    EXPECTED = {
        "input": f"question: {pad_punctuation(EXAMPLE['question'])}",
        "target": f"{pad_punctuation(EXAMPLE['answer'])}"
    }

    def test_search_qa_t5(self):
        template = PromptTemplate("search_qa")
        prompt_template = template.get_template(
            "T5", "search_qa-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
