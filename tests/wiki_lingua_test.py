import unittest
import numpy as np

from tasktemplates.template import PromptTemplate


class TestWikiLingua(unittest.TestCase):

    EXAMPLE = {
        "source_aligned": {"en": "Honesty is usually the best policy. It is disrespectful to lie to someone. If you don't want to date someone, you should say so. Sometimes it is easy to be honest. For example, you might be able to truthfully say, \"No, thank you, I already have a date for that party.\" Other times, you might need to find a kinder way to be nice. Maybe you are not attracted to the person. Instead of bluntly saying that, try saying, \"No, thank you, I just don't think we would be a good fit.\" Avoid making up a phony excuse. For instance, don't tell someone you will be out of town this weekend if you won't be. There's a chance that you might then run into them at the movies, which would definitely cause hurt feelings. A compliment sandwich is a really effective way to provide feedback. Essentially, you \"sandwich\" your negative comment between two positive things. Try using this method when you need to reject someone. An example of a compliment sandwich is to say something such as, \"You're an awesome person. Unfortunately, I'm not interested in dating you. Someone else is going to be really lucky to date someone with such a great personality!\" You could also try, \"You are a really nice person. I'm only interested you as a friend. I like when we hang out in big groups together!\" Be sincere. If you offer false compliments, the other person will likely be able to tell and feel hurt. If you do not want to date someone, it is best to be upfront about your feelings. Do not beat around the bush. If your mind is made up, it is best to clearly state your response. If someone asks you to date them and you don't want to, you can be direct and kind at the same time. State your answer clearly. You can make your feelings clear without purposefully hurting someone else's feelings. Try smiling and saying, \"That sounds fun, but no thank you. I'm not interested in dating you.\" Don't beat around the bush. If you do not want to accept the date, there is no need to say, \"Let me think about it.\" It is best to get the rejection over with. You don't want to give someone false hope. Avoid saying something like, \"Let me check my schedule and get back to you.\" Try to treat the person the way you would want to be treated. This means that you should choose your words carefully. Be thoughtful in your response. It's okay to pause before responding. You might be taken by surprise and need a moment to collect your thoughts. Say thank you. It is a compliment to be asked out. You can say, \"I'm flattered. Unfortunately, I can't accept.\" Don't laugh. Many people laugh nervously in awkward situations. Try to avoid giggling, as that is likely to result in hurt feelings. Sometimes it is not what you say, but how you say it. If you need to reject someone, think about factors other than your words. Non-verbal communication matters, too. Use the right tone of voice. Try to sound gentle but firm. Make eye contact. This helps convey that you are being serious, and also shows respect for the other person. If you are in public, try not to speak too loudly. It is not necessary for everyone around you to know that you are turning down a date."},
        "target_aligned": {"en": "Tell the truth. Use a \"compliment sandwich\". Be direct. Treat the person with respect. Communicate effectively."}
    }

    EXPECTED = {
        "input": f"{EXAMPLE['source_aligned']['en']}",
        "target": f"{EXAMPLE['target_aligned']['en']}"
    }

    def test_wiki_lingua_t5(self):
        template = PromptTemplate("wiki_lingua")
        prompt_template = template.get_template(
            "T5", "wiki_lingua-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
