import unittest

from tasktemplates.template import PromptTemplate


class TestAeslc(unittest.TestCase):

    EXAMPLE = {
        "email_body": "Greg/Phillip, Attached is the Grande Communications Service Agreement. The business points can be found in Exhibit C. I Can get the Non-Disturbance agreement after it has been executed by you and Grande. I will fill in the Legal description of the property one I have received it. Please execute and send to: Grande Communications, 401 Carlson Circle, San Marcos Texas, 78666 Attention Hunter Williams. <<Bishopscontract.doc>>",
        "subject_line": "Service Agreement"
    }

    EXPECTED = {
        "input": f"summarize: {EXAMPLE['email_body']}",
        "target": f"{EXAMPLE['subject_line']}"
    }

    def test_aeslc_t5(self):
        template = PromptTemplate("aeslc")
        prompt_template = template.get_template("T5", "aeslc-prompt-t5")

        output = prompt_template.apply(self.EXAMPLE)
        self.assertEqual(output, self.EXPECTED)


# run test
if __name__ == '__main__':
    unittest.main()
