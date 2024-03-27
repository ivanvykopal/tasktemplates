import unittest

from template import PromptTemplate
from preprocessors.core import pad_punctuation


class TestDrop(unittest.TestCase):

    EXAMPLE = {
        "question": "How many points did the buccaneers need to tie in the first?",
        "passage": "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass. The Lions won their regular season opener for the first time since 2007",
        "answers_spans": {"spans": ["3"], "types": ["number"]}
    }

    EXPECTED = {
        "input": f"question: {pad_punctuation(EXAMPLE['question'])} context: {pad_punctuation(EXAMPLE['passage'])}",
        "target": f"{pad_punctuation(EXAMPLE['answers_spans']['spans'][0])}"
    }

    def test_drop_t5(self):
        template = PromptTemplate("drop")
        prompt_template = template.get_template(
            "T5", "drop-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
