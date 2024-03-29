import unittest

from tasktemplates.template import PromptTemplate


class TestANLI(unittest.TestCase):

    EXAMPLE = {
        "uid": "0fd0abfb-659e-4453-b196-c3a64d2d8267",
        "premise": 'The Parma trolleybus system (Italian: "Rete filoviaria di Parma" ) forms part of the public transport network of the city and "comune" of Parma, in the region of Emilia-Romagna, northern Italy. In operation since 1953, the system presently comprises four urban routes.',
        "hypothesis": 'The trolleybus system has over 2 urban routes',
        "label": 0,
        "reason": None
    }

    EXPECTED = {
        "input": f"premise: {EXAMPLE['premise']} hypothesis: {EXAMPLE['hypothesis']}",
        "target": f"entailment"
    }

    def test_anli_t5(self):
        template = PromptTemplate("anli")
        prompt_template = template.get_template("T5", "anli-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
