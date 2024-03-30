import collections
import numpy as np
import re


class RecordPreprocessor():

    def preprocess(self, examples):
        new_batch = collections.defaultdict(list)
        keys = examples.keys()
        for values in zip(*examples.values()):
            ex = {k: v for k, v in zip(keys, values)}
            # updates the passage.
            passage = ex['passage']
            passage = re.sub(
                r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
            passage = re.sub(r'\n@highlight\n', '. ', passage)
            inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
            # duplicates the samples based on  number of answers.
            num_answers = len(ex["answers"])
            num_duplicates = np.maximum(1, num_answers)
            new_batch["inputs"].extend([inputs] * num_duplicates)
            new_batch["targets"].extend(
                ex["answers"] if num_answers > 0 else ["<unk>"])

            new_batch["task"].extend(['record'] * num_duplicates)
            new_batch["group"].extend(
                [ex["idx"]["query"]] * num_duplicates)

        return new_batch
