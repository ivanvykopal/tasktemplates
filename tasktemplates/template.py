import logging
import os
import yaml
from typing import Dict
import pkg_resources

from tasktemplates.utils import process_template
from tasktemplates.metrics import get_metrics
from tasktemplates.preprocessors.core import pad_punctuation, remove_markup
from tasktemplates.preprocessors.record_preprocessor import RecordPreprocessor
from tasktemplates.preprocessors.wsc_preprocessor import WSCPreprocessor

logging.basicConfig(level=logging.INFO)

TEMPLATES_FOLDER_PATH = pkg_resources.resource_filename(__name__, "templates")


class Metadata:
    """
    Metadata for the dataset
    """

    def __init__(self, metadata_dict: Dict, template_dict: Dict):
        self.languages = metadata_dict["languages"]
        self.metric_names = metadata_dict["metrics"]
        self.metrics = get_metrics(template_dict)
        self.preprocessing = metadata_dict["preprocessing"] if 'preprocessing' in metadata_dict else None


class Template:
    """
    A template for the dataset
    """

    def __init__(self, name: str, template_dict: Dict):
        self.name = name
        self.input = template_dict["input"]
        self.target = template_dict["target"]
        self.choices = template_dict["choices"] if "choices" in template_dict else None
        self.num_classes = len(self.choices) if self.choices else None
        self.metadata = Metadata(
            template_dict["metadata"], template_dict=template_dict)

    def get_preprocessing_steps(self):
        preprocessors = {
            "pad_punctuation": pad_punctuation,
            "remove_markup": remove_markup,
        }

        steps = []
        if self.metadata.preprocessing:
            for step in self.metadata.preprocessing:
                if step in preprocessors:
                    steps.append(preprocessors[step])
        return steps

    def apply_preprocessors(self, example: Dict):
        preprocessors = {
            "wsc_preprocess": WSCPreprocessor().preprocess,
            "record_preprocess": RecordPreprocessor().preprocess,
        }

        for step in self.metadata.preprocessing:
            if step in preprocessors:
                if step == "record_preprocess":
                    batched_example = {
                        key: [value] for key, value in example.items()
                    }
                    example = preprocessors[step](batched_example)
                else:
                    example = preprocessors[step](example)

        return example

    def apply(self, example: Dict):
        """
        Apply the template to the data
        """
        if self.metadata.preprocessing:
            example = self.apply_preprocessors(example)

            if 'record_preprocess' in self.metadata.preprocessing:
                return example

        input_prompt = process_template(
            self.input, example, self.choices, self.get_preprocessing_steps())

        target = process_template(
            self.target, example, self.choices, self.get_preprocessing_steps())
        return {'input': input_prompt, 'target': target}

    def get_metrics(self):
        return self.metadata.metrics


class PromptTemplate:
    """
    A prompt template
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.temp = self.load_templates_from_file()
        self.templates = self.load_templates()
        self.splits = self.temp['splits']
        self.hf_name = self.temp['dataset']

        self.subset = self.temp['subset'] if 'subset' in self.temp else None
        self.type = self.temp['type'] if 'type' in self.temp else None

    def load_templates(self):
        models = dict()
        for model_name, tems in self.temp['templates'].items():
            templates = dict()
            for template in tems:
                name = template['name']
                templates[name] = Template(name, template)
            models[model_name] = templates
        return models

    def get_template(self, model_name: str, name: str):
        """
        Get a template for the prompt
        """
        return self.templates[model_name][name]

    def load_templates_from_file(self):
        """
        Load a template from a file
        """
        template_path = os.path.join(
            TEMPLATES_FOLDER_PATH, self.dataset_name, "templates.yaml")
        with open(template_path, "r") as template_file:
            template = yaml.safe_load(template_file)
        return template

    def available_models(self):
        return list(self.temp['templates'].keys())

    def available_templates(self, model_name):
        return list(self.templates[model_name].keys())
