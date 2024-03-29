from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "datasets",
    "huggingface_hub",
    "torchmetrics",
    "scikit-learn",
    "numpy",
    "torch",
    "rouge-score",
    "sacrebleu",
]

setup(
    name='tasktemplates',
    version='0.0.6',
    url='https://github.com/ivanvykopal/tasktemplates.git',
    author='Ivan Vykopal',
    author_email='ivan.vykopal@gmail.com',
    python_requires='>=3.6.0',
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    description="Task templates for NLP datasets",
    packages=find_packages(),
    license='MIT',
    long_description=readme,
    long_description_content_type="text/markdown",
    package_data={'': ['templates/*/*.yaml']},
)
