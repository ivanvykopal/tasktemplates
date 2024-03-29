import argparse
import os
import yaml
import pkg_resources

from tasktemplates.dataset import Dataset
from tasktemplates.template import PromptTemplate

TEMPLATES_FOLDER_PATH = pkg_resources.resource_filename(__name__, "templates")


class TemplateCheck:
    """
    Check whether the yaml file is valid and whether the templates contains all the necessary fields
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.temp = self.load_templates_from_file()
        self.check_template()

    def load_templates_from_file(self):
        """
        Load the templates from the yaml file
        """
        with open(f"{TEMPLATES_FOLDER_PATH}/{self.dataset_name}/templates.yaml") as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    def check_template(self):
        """
        Check if the template is valid
        """
        assert 'dataset' in self.temp, f"The dataset name is missing in {self.dataset_name}"
        assert 'splits' in self.temp, f"The splits are missing in {self.dataset_name}"
        assert 'templates' in self.temp, f"The templates are missing in {self.dataset_name}"

        assert 'train' in self.temp[
            'splits'], f"The train split is missing in {self.dataset_name}"
        assert 'validation' in self.temp[
            'splits'], f"The validation split is missing in {self.dataset_name}"
        assert 'test' in self.temp[
            'splits'], f"The test split is missing in {self.dataset_name}"

        # templates should contain the dictionary of models with the templates
        for _, templates in self.temp['templates'].items():
            for template in templates:
                assert 'name' in template, f"The name of the template is missing in {self.dataset_name}"
                assert 'input' in template, f"The input is missing in {self.dataset_name}"
                assert 'target' in template, f"The target is missing in {self.dataset_name}"
                assert 'metadata' in template, f"The metadata is missing in {self.dataset_name}"
                assert 'languages' in template[
                    'metadata'], f"The languages are missing in {self.dataset_name}"
                assert 'metrics' in template[
                    'metadata'], f"The metrics are missing in {self.dataset_name}"

        return True

    def check_prompt_templates(self):
        template = PromptTemplate(self.dataset_name)
        models = template.available_models()
        for model in models:
            prompt_templates = template.available_templates(model)
            for prompt in prompt_templates:
                print(f"Checking template {prompt} for model {model}")
                dataset = Dataset(template, model_name=model,
                                  prompt_name=prompt)
                example = dataset.dataset[dataset.splits['train']][0]
                prompt_func = template.get_template(model, prompt)

                try:
                    prompt_func.apply(example)
                except Exception as e:
                    raise Exception(
                        f"Error in applying the template {prompt} for model {model}: {e}")

        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-templates',
                        action='store_true', help='Check the templates')
    parser.add_argument("--dataset", type=str,
                        help="The dataset to check", default=None)
    args = parser.parse_args()

    if args.check_templates:
        # check all yaml files in templates folder
        datasets = os.listdir("templates")
        for dataset in datasets:
            TemplateCheck(dataset)
    elif args.dataset is not None:
        TemplateCheck(args.dataset).check_prompt_templates()
    else:
        raise ValueError("Please provide a dataset to check")
