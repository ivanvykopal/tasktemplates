import unittest

from template import PromptTemplate


class TestSamSum(unittest.TestCase):

    EXAMPLE = {
        "dialogue": "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)",
        "summary": "Amanda baked cookies and will bring Jerry some tomorrow."
    }

    EXPECTED = {
        "input": f"summarize: {EXAMPLE['dialogue']}",
        "target": f"{EXAMPLE['summary']}"
    }

    def test_samsum_t5(self):
        template = PromptTemplate("samsum")
        prompt_template = template.get_template(
            "T5", "samsum-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
