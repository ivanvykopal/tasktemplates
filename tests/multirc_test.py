import unittest

from tasktemplates.template import PromptTemplate

# TODO: Update the test case with the preprocessing step


class TestMultiRC(unittest.TestCase):

    EXAMPLE = {
        "question": "What did the high-level effort to persuade Pakistan include?",
        "answer": "Children, Gerd, or Dorian Popa",
        "paragraph": "While this process moved along, diplomacy continued its rounds. Direct pressure on the Taliban had proved unsuccessful. As one NSC staff note put it, \"Under the Taliban, Afghanistan is not so much a state sponsor of terrorism as it is a state sponsored by terrorists.\" In early 2000, the United States began a high-level effort to persuade Pakistan to use its influence over the Taliban. In January 2000, Assistant Secretary of State Karl Inderfurth and the State Department's counterterrorism coordinator, Michael Sheehan, met with General Musharraf in Islamabad, dangling before him the possibility of a presidential visit in March as a reward for Pakistani cooperation. Such a visit was coveted by Musharraf, partly as a sign of his government's legitimacy. He told the two envoys that he would meet with Mullah Omar and press him on Bin Laden. They left, however, reporting to Washington that Pakistan was unlikely in fact to do anything,\" given what it sees as the benefits of Taliban control of Afghanistan.\" President Clinton was scheduled to travel to India. The State Department felt that he should not visit India without also visiting Pakistan. The Secret Service and the CIA, however, warned in the strongest terms that visiting Pakistan would risk the President's life. Counterterrorism officials also argued that Pakistan had not done enough to merit a presidential visit. But President Clinton insisted on including Pakistan in the itinerary for his trip to South Asia. His one-day stopover on March 25, 2000, was the first time a U.S. president had been there since 1969. At his meeting with Musharraf and others, President Clinton concentrated on tensions between Pakistan and India and the dangers of nuclear proliferation, but also discussed Bin Laden. President Clinton told us that when he pulled Musharraf aside for a brief, one-on-one meeting, he pleaded with the general for help regarding Bin Laden.\" I offered him the moon when I went to see him, in terms of better relations with the United States, if he'd help us get Bin Laden and deal with another issue or two.\" The U.S. effort continued.",
        "label": 0
    }

    EXPECTED = {
        "input": f"multirc question: {EXAMPLE['question']} answer: {EXAMPLE['answer']} paragraph: {EXAMPLE['paragraph']}",
        "target": f"False"
    }

    def test_multirc_t5(self):
        template = PromptTemplate("multirc")
        prompt_template = template.get_template(
            "T5", "multirc-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
