import unittest

from template import PromptTemplate


class TestBoolQ(unittest.TestCase):

    EXAMPLE = {
        "question": "do iran and afghanistan speak the same language",
        "passage": "Persian language -- Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.",
        "label": 1
    }

    EXPECTED = {
        "input": f"question: {EXAMPLE['question']} passage: {EXAMPLE['passage']}",
        "target": f"True"
    }

    def test_boolq_t5(self):
        template = PromptTemplate("boolq")
        prompt_template = template.get_template("T5", "boolq-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
