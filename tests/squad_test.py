import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.core import pad_punctuation


class TestSquad(unittest.TestCase):

    EXAMPLE = {
        "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
        "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
        "answers": {"text": ["Saint Bernadette Soubirous"], "answer_start": [515]}
    }

    EXPECTED = {
        "input": f"question: {pad_punctuation(EXAMPLE['question'])} context: {pad_punctuation(EXAMPLE['context'])}",
        "target": f"{pad_punctuation(EXAMPLE['answers']['text'][0])}"
    }

    EXPECTED_TRIVIA = {
        "input": f"squad trivia question: {pad_punctuation(EXAMPLE['question'])}",
        "target": f"{pad_punctuation(EXAMPLE['answers']['text'][0])}"
    }

    def test_squad_t5(self):
        template = PromptTemplate("squad")
        prompt_template = template.get_template(
            "T5", "squad-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)

    def test_squad_t5_2(self):
        template = PromptTemplate("squad")
        prompt_template = template.get_template(
            "T5", "squad-prompt-trivia-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED_TRIVIA)


# run test
if __name__ == '__main__':
    unittest.main()
