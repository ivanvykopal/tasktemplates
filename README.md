# TaskTemplates 

## Prompt Templates

To use prompt form our templates import `PromptTemplate` class from `template`:

Usage:
```
from template import PromptTemplate

template = PromptTemplate('multirc')
multirc_prompt = template.get_template('T5', ''multirc-prompt-t5')
```

To load dataset use `Dataset` from `dataset`:

Usage:
```
from dataset import Dataset

dataset = Dataset('multirc', 'T5', 'multirc-prompt-t5')
```