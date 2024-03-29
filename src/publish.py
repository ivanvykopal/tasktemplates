import argparse
import os
import yaml
import json
from huggingface_hub import HfApi


def create_json(hf_token: str):
    """
    Create a jsonl file with the dataset
    """
    api = HfApi()
    data = []
    datasets = os.listdir('src/templates')
    for dataset in datasets:
        print(f"Checking dataset {dataset}")
        path = f'src/templates/{dataset}/templates.yaml'
        with open(path) as file:
            templates = yaml.load(file, Loader=yaml.FullLoader)
            for model, templates in templates['templates'].items():
                for template in templates:
                    data.append({
                        'dataset': dataset,
                        'model': model,
                        'name': template['name'],
                        'input': template['input'],
                        'target': template['target'],
                        'metadata': template['metadata'],
                        'languages': template['metadata']['languages'],
                        'metrics': template['metadata']['metrics']
                    })

    with open('src/dataset/train.jsonl', 'w') as file:
        for line in data:
            file.write(json.dumps(line))
            file.write('\n')

    # publish it to the huggingface hub
    api.upload_file(
        token=hf_token,
        path_or_fileobj='src/dataset/train.jsonl',
        path_in_repo='train.jsonl',
        repo_id='ivykopal/tasktemplates',
        repo_type='dataset',
    )


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--hf-token', type=str, required=True)
    args = argparser.parse_args()
    create_json(hf_token=args.hf_token)
