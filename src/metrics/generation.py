from torchmetrics import Metric
from rouge_score import rouge_scorer
from rouge_score import scoring
import sacrebleu
import numpy as np
import collections
import re

from metrics import qa_utils
from metrics.utils import prepare_summary_rouge


class BLEU(Metric):
    def __init__(self, tokenizer="intl"):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        if isinstance(targets[0], list):
            targets = [[x for x in target] for target in targets]
        else:
            targets = [targets]

        self.preds += preds
        self.targets += targets

    def compute(self):
        return sacrebleu.corpus_bleu(
            self.preds, self.targets,
            smooth_method="exp",
            smooth_value=0.0,
            force=False,
            lowercase=False,
            tokenize=self.tokenizer,
            use_effective_order=False
        ).score


class Rouge(Metric):
    def __init__(self, rouge_types=["rouge1", "rouge2", "rougeLsum"]):
        super().__init__()
        self.rouge_types = rouge_types
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        self.preds += preds
        self.targets += targets

    def compute(self):
        scorer = rouge_scorer.RougeScorer(self.rouge_types)
        aggregator = scoring.BootstrapAggregator()

        for target, pred in zip(self.targets, self.preds):
            target = prepare_summary_rouge(target)
            pred = prepare_summary_rouge(pred)
            aggregator.add_scores(scorer.score(target=target, prediction=pred))

        result = aggregator.aggregate()
        return {key: value.mid.fmeasure * 100 for key, value in result.items()}


class RougeMean(Metric):
    def __init__(self, rouge_types=["rouge1", "rouge2", "rougeLsum"]):
        super().__init__()
        self.rouge_types = rouge_types
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        self.preds += preds
        self.targets += targets

    def compute(self):
        scorer = rouge_scorer.RougeScorer(rouge_types=self.rouge_types)
        count = 0
        sum_scores = collections.defaultdict(float)
        for prediction, target in zip(self.preds, self.targets):
            target = prepare_summary_rouge(target)
            prediction = prepare_summary_rouge(prediction)
            scores = scorer.score(target=target, prediction=prediction)
            count += 1
            for k, v in scores.items():
                sum_scores[k] += v.fmeasure
        if count == 0:
            raise ValueError(
                "Predictions and targets must both have nonzero length")
        result = {k: v / count for k, v in sum_scores.items()}
        return {key: result[key] * 100 for key in self.rouge_types}


class Squad(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        if type(targets[0]) is list:
            targets = [[qa_utils.normalize_squad(
                t) for t in u] for u in targets]
        else:
            targets = [[qa_utils.normalize_squad(u)] for u in targets]

        preds = [qa_utils.normalize_squad(p) for p in preds]

        self.preds += preds
        self.targets += targets

    def compute(self):
        return qa_utils.qa_metrics(targets, predictions)


class SpanSquad(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.squad = Squad()

    def update(self, preds, targets):
        assert len(targets) == len(preds)

        self.preds += preds
        self.targets += targets

    def compute(self):

        def space_tok(s):
            return re.sub(r"\W", " ", s).split()

        def get_answer_text_from_context(context, answer_tokens):
            """Find the answer in the context given the answer tokens."""
            # In the initial training iterations, the model can output garbage.
            # Returning an empty string in such cases.
            if len(answer_tokens) < 4:
                return ""

            # Model sometimes predicts words instead of numbers in the answer. Return
            # an empty string in that case.
            try:
                start_index = int(answer_tokens[1])
                end_index = int(answer_tokens[3])
            except ValueError:
                return ""

            return " ".join(context[start_index:end_index+1])

        contexts = [space_tok(t["context"]) for t in self.targets]
        answers = [t["answers"] for t in self.targets]

        predictions = [space_tok(p) for p in self.preds]
        final_predictions = [
            get_answer_text_from_context(c, p) for c, p in zip(contexts, predictions)
        ]

        return self.squad(answers, final_predictions)


class TriviaQA(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        assert len(targets) == len(preds)

        self.preds += preds
        self.targets += targets

    def compute(self):
        targets = [[qa_utils.normalize_trivia_qa(
            t) for t in u] for u in self.targets]
        predictions = [qa_utils.normalize_trivia_qa(p) for p in self.preds]
        return qa_utils.qa_metrics(targets, predictions)


class ExactMatch(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        assert len(targets) == len(preds)

        self.preds += preds
        self.targets += targets

    def compute(self):
        return 100 * float(np.array_equal(targets, predictions))
