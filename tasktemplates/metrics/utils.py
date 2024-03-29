import sklearn.metrics


def sklearn_metrics_wrapper(metric_str,
                            metric_dict_str=None,
                            metric_post_process_fn=None,
                            **metric_fn_kwargs):
    """Wraps any sklearn.metric function and returns a t5 metric function.

    Args:
      metric_str: string, the function from `sklearn.metrics` to use.
      metric_dict_str: optional string, if not specified `metric_str` is used as
        the key in the returned dictionary.
      metric_post_process_fn: callable, if specified the final computed metric
        will be passed through this.
      **metric_fn_kwargs: kwargs, passed to the metric function we are calling.

    Returns:
      the function that calculates the metric in a dict.
    """
    if not hasattr(sklearn.metrics, metric_str):
        raise ValueError("sklearn.metrics does not have: %s" % metric_str)

    def fn(targets, predictions):
        metric_fn = getattr(sklearn.metrics, metric_str)
        metric_val = metric_fn(targets, predictions, **metric_fn_kwargs)
        if metric_post_process_fn is not None:
            metric_val = metric_post_process_fn(metric_val)
        return {metric_dict_str or metric_str: metric_val}
    return fn


def prepare_summary_rouge(summary):
    # Make sure the summary is not bytes-type
    # Add newlines between sentences so that rougeLsum is computed correctly.
    summary = summary.replace(" . ", " .\n")
    return summary


def tags_to_spans(tag_sequence, delimiter=', '):
    """Extract spans from IOB1 or BIO tags."""
    tag_sequence_split = [x.strip() for x in tag_sequence.split(delimiter)]
    tags_entities = []
    for tag_entity in tag_sequence_split:
        tag_entity_split = tag_entity.split(':')
        if len(tag_entity_split) != 2:
            continue
        tag = tag_entity_split[0].strip()
        entity = tag_entity_split[1].strip()
        tags_entities.append((tag, entity))
    return tags_entities


def compute_f1_metrics(true_positives, false_positives, false_negatives):
    precision = float(true_positives) / float(true_positives + false_positives +
                                              1e-13)
    recall = float(true_positives) / float(true_positives + false_negatives +
                                           1e-13)
    f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
    return precision, recall, f1_measure
