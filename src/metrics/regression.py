from torchmetrics import Metric
import scipy.stats
import numpy as np


class PearsonCorrelation(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        self.preds += preds
        self.targets += targets

    def compute(self):
        targets = np.asarray(targets, dtype=np.float16)
        predictions = np.asarray(predictions, dtype=np.float16)
        return 100 * scipy.stats.pearsonr(targets, predictions)[0]


class SpearmanCorrelation(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        self.preds += preds
        self.targets += targets

    def compute(self):
        targets = np.asarray(targets, dtype=np.float16)
        predictions = np.asarray(predictions, dtype=np.float16)
        return 100 * scipy.stats.spearmanr(targets, predictions)[0]
