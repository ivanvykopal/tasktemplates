from torchmetrics import Metric
import sklearn.metrics
import numpy as np
import torch
from torchmetrics.functional import confusion_matrix
from sklearn.metrics import f1_score
import collections

from metrics.utils import sklearn_metrics_wrapper, tags_to_spans, compute_f1_metrics


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        self.preds += preds
        self.targets += targets

    def compute(self):
        return sklearn.metrics.accuracy_score(self.targets, self.preds) * 100


class SequenceAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        assert len(preds) == len(targets)
        self.preds += preds
        self.targets += targets

    def compute(self):
        return 100 * np.mean([p == t for p, t in zip(self.preds, self.targets)])


class MultiClassF1(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.num_classes = num_classes

    def update(self, preds, targets):
        assert len(preds) == len(targets)
        self.preds += preds
        self.targets += targets

    def compute(self):
        return sklearn_metrics_wrapper(
            "fbeta_score",
            metric_dict_str="mean_%dclass_f1" % self.num_classes,
            metric_post_process_fn=lambda x: 100 * x,
            beta=1,
            labels=range(self.num_classes),
            average="macro"
        )


class F1Invalid(Metric):  # Need to fix this
    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def binary_reverse(targets):
        return ["0" if target == "1" else "1" for target in targets]

    def update(self, preds, targets):
        assert len(preds) == len(targets)
        targets, preds = np.asarray(targets), np.asarray(preds)

        invalid_idx_mask = np.logical_and(preds != 0, preds != 1)
        # preds[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
        preds[invalid_idx_mask] = self.binary_reverse(
            targets[invalid_idx_mask])

        preds, targets = torch.tensor(preds.astype(np.int32)), torch.tensor(
            targets.astype(np.int32)
        )

        conf_mat = confusion_matrix(
            preds, targets, num_classes=2, task="binary")
        self.tn += conf_mat[0, 0]
        self.fp += conf_mat[0, 1]
        self.fn += conf_mat[1, 0]
        self.tp += conf_mat[1, 1]

    def compute(self):
        if self.tp + self.fp == 0:
            return torch.tensor(0.0)

        if self.tp + self.fn == 0:
            return torch.tensor(0.0)

        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)

        if (precision * recall) == 0:
            return torch.tensor(0.0)

        if (precision + recall) == 0:
            return torch.tensor(0.0)

        return 100 * 2 * (precision * recall) / (precision + recall)


class MultiRCF1(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

        self.f1_score_with_invalid = F1Invalid()

    def update(self, preds, targets):
        assert len(preds) == len(targets)
        self.preds += preds
        self.targets += targets

    def compute(self):
        return self.f1_score_with_invalid(
            [p["value"] for p in self.preds], [t["value"]
                                               for t in self.targets]
        )


class AUC(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        assert len(preds) == len(targets)
        self.preds += preds
        self.targets += targets

    def compute(self):
        return sklearn.metrics.roc_auc_score(self.targets, self.preds) * 100


class MacroF1(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.num_classes = num_classes

    def update(self, preds, targets):
        assert len(preds) == len(targets)
        self.preds += preds
        self.targets += targets

    def compute(self):
        return f1_score(self.targets, self.preds, average="macro") * 100


class SpanF1(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        assert len(preds) == len(targets)
        self.preds += preds
        self.targets += targets

    def compute(self):
        true_positives = collections.defaultdict(int)
        false_positives = collections.defaultdict(int)
        false_negatives = collections.defaultdict(int)

        for target, pred in zip(targets, predictions):
            gold_spans = tags_to_spans(target)
            predicted_spans = tags_to_spans(pred)

            for span in predicted_spans:
                if span in gold_spans:
                    true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    false_positives[span[0]] += 1
            # These spans weren't predicted.
            for span in gold_spans:
                false_negatives[span[0]] += 1

        _, _, f1_measure = compute_f1_metrics(
            sum(true_positives.values()), sum(false_positives.values()),
            sum(false_negatives.values()))

        return f1_measure * 100
