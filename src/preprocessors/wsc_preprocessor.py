from preprocessors.core import mark_span


class WSCPreprocessor():

    def preprocess(self, example):
        """Convert WSC examples to text2text format.

        For example, a typical example from WSC might look like
        {
            'text': 'This is a test sentence .',
            'span1_text': 'test',
            'span1_index': 3,
            'span2_text': 'This',
            'span2_index': 0,
            'label': 0
        }

        This example would be transformed to
        {
            'inputs': 'wsc text: # This # is a * test * sentence .',
            'targets': 'False'
        }

        Args:
        x: an example to process.
        Returns:
        A preprocessed example.
        """
        text = example['text']
        text = mark_span(
            text, example['span1_text'], example['span1_index'], '*')
        # Compensate for 2 added "words" added in previous step.
        span2_index = example['span2_index'] + 2 * \
            int(example['span1_index'] < example['span2_index'])
        text = mark_span(text, example['span2_text'], span2_index, '#')
        return {
            "text": text,
            "label": example['label']
        }
