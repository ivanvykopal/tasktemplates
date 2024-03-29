import unittest
import numpy as np

from tasktemplates.template import PromptTemplate


class TestWikiAuto(unittest.TestCase):

    EXAMPLE = {
        "source": "	Pterocarpus indicus ( commonly known as Amboyna wood , Malay padauk , Papua New Guinea rosewood , Philippine mahogany , Andaman redwood , Burmese rosewood , narra , angsana , or Pashu padauk ) is a species of \" Pterocarpus \" native to southeastern Asia , northern Australasia , and the western Pacific Ocean islands , in Cambodia , southernmost China , East Timor , Indonesia , Malaysia , Papua New Guinea , the Philippines , the Ryukyu Islands , the Solomon Islands , Thailand , and Vietnam .",
        "target": "Pterocarpus indicus ( commonly known as Amboyna wood , Malay padauk , Papua New Guinea rosewood , Philippine mahogany , Andaman redwood , Burmese rosewood , narra or Pashu padauk ) is a species of \" Pterocarpus \" native to southeastern Asia ."
    }

    EXPECTED = {
        "input": f"{EXAMPLE['source']}",
        "target": f"{EXAMPLE['target']}"
    }

    def test_wiki_auto_t5(self):
        template = PromptTemplate("wiki_auto")
        prompt_template = template.get_template(
            "T5", "wiki_auto-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
