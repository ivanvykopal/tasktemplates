from metrics.classification import AUC, Accuracy, F1Invalid, MacroF1, MultiClassF1, MultiRCF1, SequenceAccuracy, SpanF1
from metrics.generation import BLEU, ExactMatch, Rouge, RougeMean, SpanSquad, Squad, TriviaQA
from metrics.regression import PearsonCorrelation, SpearmanCorrelation
from template import PromptTemplate


def get_metrics(template: PromptTemplate):
    """
    Get the metrics for the dataset
    """
    metrics = []
    deined_metrics = {
        'accuracy': Accuracy(),
        'sequqnce_accuracy': SequenceAccuracy(),
        'multiclass_f1': MultiClassF1(template.num_classes),
        'f1_invalid': F1Invalid(),
        'multirc_f1': MultiRCF1(),
        'auc': AUC(),
        'macro_f1': MacroF1(template.num_classes),
        'span_f1': SpanF1(),
        'bleu': BLEU(),
        'rouge': Rouge(),
        'rouge_mean': RougeMean(),
        'squad': Squad(),
        'span_squad': SpanSquad(),
        'trivia_qa': TriviaQA(),
        'exact_match': ExactMatch(),
        'pearson_corref': PearsonCorrelation(),
        'corref': SpearmanCorrelation(),
    }
    for metric in template.metadata.metrics:
        metrics.append(deined_metrics[metric])

    return metrics
