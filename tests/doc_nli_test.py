import unittest

from tasktemplates.template import PromptTemplate


class TestDocNLI(unittest.TestCase):

    EXAMPLE = {
        "premise": "The Parma trolleybus system (Italian: \"Rete filoviaria di Parma\" ) forms part of the public transport network of the city and \"comune\" of Parma, in the region of Emilia-Romagna, northern Italy. In operation since 1953, the system presently comprises four urban routes.",
        "hypothesis": "The trolleybus system has over 2 urban routes",
        "label": "entailment"
    }
    EXPECTED = {
        "input": f"premise: {EXAMPLE['premise']} hypothesis: {EXAMPLE['hypothesis']}",
        "target": f"{EXAMPLE['label']}"
    }

    def test_doc_nli_t5(self):
        template = PromptTemplate("doc_nli")
        prompt_template = template.get_template(
            "T5", "doc_nli-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
