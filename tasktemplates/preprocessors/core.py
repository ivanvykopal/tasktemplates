import regex
import re


def pad_punctuation(text):
    """Adds spaces around punctuation."""
    text = regex.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', str(text))
    # Collapse consecutive whitespace into one space.
    text = regex.sub(r'\s+', ' ', text)
    return text


def remove_markup(text):
    """Removes the HTML markup."""
    text = re.sub('<br>', ' ', text)
    text = re.sub('<(/)?b>', '', text)
    return text


def mark_span(text, span_str, span_idx, mark):
    pattern_tmpl = r'^((?:\S+\s){N})(W)'
    pattern = re.sub('N', str(span_idx), pattern_tmpl)
    pattern = re.sub('W', span_str, pattern)
    return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

# record_preprocess
