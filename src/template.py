import logging
import os
import yaml

from utils import process_template

logging.basicConfig(level=logging.INFO)


class Metadata:
    """
    Metadata for the dataset
    """

    def __init__(self, metadata_dict):
        self.languages = metadata_dict["languages"]
        self.metrics = metadata_dict["metrics"]
        self.preprocessing = metadata_dict["preprocessing"] if 'preprocessing' in metadata_dict else None


class Template:
    """
    A template for the dataset
    """

    def __init__(self, name, template_dict):
        self.name = name
        self.input = template_dict["input"]
        self.target = template_dict["target"]
        self.metadata = Metadata(template_dict["metadata"])

        self.choices = template_dict["choices"] if "choices" in template_dict else None
        self.num_classes = len(self.choices) if self.choices else None

    def apply(self, example):
        """
        Apply the template to the data
        """
        input_prompt = process_template(self.input, example, self.choices)

        target = process_template(self.target, example, self.choices)
        return {'input': input_prompt, 'target': target}


class PromptTemplate:
    """
    A prompt template
    """

    def __init__(self, dataset_name):
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

    def get_template(self, model_name, name):
        """
        Get a template for the prompt
        """
        return self.templates[model_name][name]

    def load_templates_from_file(self):
        """
        Load a template from a file
        """
        template_path = os.path.join(
            ".", "templates", self.dataset_name, "templates.yaml")
        with open(template_path, "r") as template_file:
            template = yaml.safe_load(template_file)
        return template

    def available_models(self):
        return list(self.temp['templates'].keys())
