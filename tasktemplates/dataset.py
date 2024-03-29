from tasktemplates.template import PromptTemplate
from datasets import load_dataset
from typing import Union, Dict, Any
import logging
import functools

logging.basicConfig(level=logging.INFO)


class Dataset:
    def __init__(
            self,
            template: Union[PromptTemplate, str],
            model_name: str = None,
            prompt_name: str = None
    ):
        if isinstance(template, str):
            template = PromptTemplate(template)

        self.dataset_name = template.hf_name
        self.subset = template.subset
        self.splits = template.splits
        self.preprocessed = False
        self._load_dataset()
        self.prompt_template = template.get_template(model_name, prompt_name)

    def _load_dataset(self):
        self.dataset = load_dataset(self.dataset_name, self.subset)
        logging.info(
            f"Loaded dataset {self.dataset_name} with subset {self.subset}")

    def tokenize(
        self,
        example: Dict,
        tokenizer: Any,
        max_input_length: int,
        max_target_length: int,
        padding: str = "max_length",
        truncation: bool = True,
        pad_token: int = -100
    ):
        input = example["input"]
        target = example["target"]

        inputs = tokenizer(input, max_length=max_input_length,
                           padding=padding, truncation=truncation, return_tensors="pt")
        targets = tokenizer(target, max_length=max_target_length,
                            padding=padding, truncation=truncation, return_tensors="pt")

        if pad_token is not None:
            targets = targets["input_ids"]
            targets[targets == tokenizer.pad_token_id] = pad_token

        inputs["labels"] = targets

        return inputs

    def tokenize_dataset(
        self,
        tokenizer: Any,
        max_input_length: int,
        max_target_length: int,
        padding: str = "max_length",
        truncation: bool = True,
        pad_token: int = -100
    ):
        if not self.preprocessed:
            self.preprocess()

        func = functools.partial(
            self.tokenize,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            padding=padding,
            truncation=truncation,
            pad_token=pad_token
        )
        train_data = self.dataset[self.splits['train']].map(
            func, batched=True, load_from_cache_file=False)
        val_data = self.dataset[self.splits['validation']].map(
            func, batched=True, load_from_cache_file=False)
        test_data = self.dataset[self.splits['test']].map(
            func, batched=True, load_from_cache_file=False)

        return train_data, val_data, test_data

    def preprocess(self, batched: bool = True, remove_columns: bool = True):
        # preprocess using self.prompt_template
        list_remove_columns = []
        if remove_columns:
            list_remove_columns = self.dataset[self.splits['train']].column_names

        if batched:
            def apply_fn(batch):
                inputs = []
                targets = []
                keys = list(batch.keys())
                length = len(batch[keys[0]])
                for i in range(length):
                    example = self.prompt_template.apply({
                        key: batch[key][i] for key in keys
                    })
                    inputs.append(example['input'])
                    targets.append(example['target'])

                return {'input': inputs, 'target': targets}

            self.dataset = self.dataset.map(
                apply_fn, batched=True, remove_columns=list_remove_columns)
            self.preprocessed = True
        else:
            self.dataset = self.dataset.map(
                self.prompt_template.apply, batched=batched, remove_columns=list_remove_columns)
            self.preprocessed = True
