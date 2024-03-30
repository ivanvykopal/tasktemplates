import unittest

from tasktemplates.template import PromptTemplate
from tasktemplates.preprocessors.record_preprocessor import RecordPreprocessor


class TestRecord(unittest.TestCase):

    EXAMPLE = {
        "passage": "The harrowing stories of women and children locked up for so-called 'moral crimes' in Afghanistan's notorious female prison have been revealed after cameras were allowed inside. Mariam has been in Badam Bagh prison for three months after she shot a man who just raped her at gunpoint and then turned the weapon on herself - but she has yet to been charged. Nuria has eight months left to serve of her sentence for trying to divorce her husband. She gave birth in prison to her son and they share a cell together. Scroll down for video Nuria was jailed for trying to divorce her husband. Her son is one of 62 children living at Badam Bagh prison @highlight Most of the 202 Badam Bagh inmates are jailed for so-called 'moral crimes' @highlight Crimes include leaving their husbands or refusing an arrange marriage @highlight 62 children live there and share cells with their mothers and five others",
        "query": "The baby she gave birth to is her husbands and he has even offered to have the courts set her free if she returns, but @placeholder has refused.",
        "entities": ["Badam Bagh", "Nuria", "Mariam", "Afghanistan"],
        "entity_spans": {"text": ["Afghanistan", "Mariam", "Badam Bagh", "Nuria", "Nuria", "Badam Bagh", "Badam Bagh"], "start": [86, 178, 197, 357, 535, 627, 672], "end": [97, 184, 207, 362, 540, 637, 682]},
        "answers": ["Nuria"],
        "idx": {"passage": 0, "query": 0}
    }

    EXPECTED = {
        "input": f"record query: {EXAMPLE['query']} entities: {', '.join(EXAMPLE['entities'])} passage: {EXAMPLE['passage']}",
        "target": f"{EXAMPLE['answers'][0]}"
    }

    def test_record_t5(self):
        template = PromptTemplate("record")
        prompt_template = template.get_template(
            "T5", "record-prompt-t5")

        # output = prompt_template.apply(self.EXAMPLE)
        # self.assertEqual(output, self.EXPECTED)
        # convert to btached example
        batched_example = {
            "passage": [self.EXAMPLE["passage"]],
            "query": [self.EXAMPLE["query"]],
            "entities": [self.EXAMPLE["entities"]],
            "entity_spans": [self.EXAMPLE["entity_spans"]],
            "answers": [self.EXAMPLE["answers"]],
            "idx": [self.EXAMPLE["idx"]]
        }

        expected = RecordPreprocessor().preprocess(batched_example)
        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(expected, output)


# run test
if __name__ == '__main__':
    unittest.main()
