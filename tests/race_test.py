import unittest

from tasktemplates.template import PromptTemplate


class TestRace(unittest.TestCase):

    EXAMPLE = {
        "question": "Bae Seul-Ki _ in the MV of the song according to the passage.",
        "article": "\"I planted a seed. Finally grow fruits. Today is a great day. Pick off the star for you. Pick off the moon for you. Let it rise for you every day. Become candles burning myself. Just light you up, hey!... You are my little little apple. How much I love you, still no enough.\" This words are from the popular song You Are My Little Dear Apple. Bae Seul-Ki acted as the leading dancer in the MV of the song. She loves dancing. She became crazy about hip-hop when she was a school girl. Bai Seul-Ki was born on September 27, 1986. She is a South Korean singer and dancer. She is 168cm tall. She loves cooking. Her favourite food is spicy and salty. She like pink and red most. There are five members in her family---father, mother, two younger brothers and herself. She isn't married. After her father and mother broke up, she lived with her mother and new daddy. She enjoys being alone.",
        "answer": "B",
        "options": ["sang", "danced", "cried", "laughed"]
    }

    EXPECTED = {
        "input": f"question: {EXAMPLE['question']} context: {EXAMPLE['article']} choice0: {EXAMPLE['options'][0]} choice1: {EXAMPLE['options'][1]} choice2: {EXAMPLE['options'][2]} choice3: {EXAMPLE['options'][3]}",
        "target": f"1"
    }

    def test_race_t5(self):
        template = PromptTemplate("race")
        prompt_template = template.get_template(
            "T5", "race-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
