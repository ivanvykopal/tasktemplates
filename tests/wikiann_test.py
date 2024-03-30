import unittest

from tasktemplates.template import PromptTemplate


class TestWikiANN(unittest.TestCase):

    EXAMPLE = {
        "tokens": ["R.H.", "Saunders", "(", "St.", "Lawrence", "River", ")", "(", "968", "MW", ")"],
        "ner_tags": [3, 4, 0, 3, 4, 4, 0, 0, 0, 0, 0],
        "langs": ["en", "en", "en", "en", "en", "en", "en", "en", "en", "en", "en"],
        "spans": ["ORG: R.H. Saunders", "ORG: St. Lawrence River"],
    }

    EXPECTED = {
        "input": f"Sentence: {' '.join(EXAMPLE['tokens'])}\\nIdentify all named entities in the sentence using PER, LOC, ORG.",
        "target": f"{', '.join(EXAMPLE['spans'])}"
    }

    def test_wikiann_mt0(self):
        template = PromptTemplate("wikiann")
        prompt_template = template.get_template(
            "mT0", "wikiann-prompt-mt0")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
