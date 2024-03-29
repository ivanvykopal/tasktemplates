from tasktemplates.metrics.classification import AUC, Accuracy, F1Invalid, MacroF1, MultiClassF1, MultiRCF1, SequenceAccuracy, SpanF1
from tasktemplates.metrics.generation import BLEU, ExactMatch, Rouge, RougeMean, SpanSquad, Squad, TriviaQA
from tasktemplates.metrics.regression import MatthewsCorrelation, PearsonCorrelation, SpearmanCorrelation


def get_metrics(template):
    """
    Get the metrics for the dataset
    """
    metrics = []
    num_classes = len(template['choices']) if 'choices' in template else None
    deined_metrics = {
        'accuracy': Accuracy(),
        'sequence_accuracy': SequenceAccuracy(),
        'multiclass_f1': MultiClassF1(num_classes),
        'f1_invalid': F1Invalid(),
        'multirc_f1': MultiRCF1(),
        'auc': AUC(),
        'macro_f1': MacroF1(num_classes),
        'span_f1': SpanF1(),
        'bleu': BLEU(),
        'rouge': Rouge(),
        'rouge_mean': RougeMean(),
        'squad': Squad(),
        'span_squad': SpanSquad(),
        'trivia_qa': TriviaQA(),
        'exact_match': ExactMatch(),
        'pearson_corrcoef': PearsonCorrelation(),
        'spearman_corrcoef': SpearmanCorrelation(),
        'matthews_corrcoef': MatthewsCorrelation(),
    }
    for metric in template['metadata']['metrics']:
        metrics.append(deined_metrics[metric])

    return metrics
