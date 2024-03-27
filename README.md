# Intent Classification on MASSIVE Dataset

## Abstract
- **Pre-trained Models**: Utilization of BERT for intent classification on MASSIVE dataset.
- **Advanced Techniques**: Exploration of learning rate schedulers, warmup steps, and contrastive learning.
- **Performance Evaluation**: Effectiveness measured by accuracy metric differences.

## Introduction
- **Dataset**: MASSIVE dataset by Amazon with 60 unique intent classes.
- **Objective**: Develop a model for effective intent classification.

## Methods
- **Data Processing**: Tokenization of text for BERT model input.
- **Model Training**: Baseline IntentModel with hyperparameter tuning and fine-tuning techniques.

## Results
- **Model Comparison**: Evaluation of baseline, custom, and contrastive learning models.
- **Accuracy Improvement**: Significant increases with advanced techniques.

## Discussion
- **Techniques Impact**: Analysis of individual and combined effects of techniques on model accuracy.
- **Further Improvements**: Suggestions for enhancing model performance.

## Requirements
We are using [poetry](https://python-poetry.org/) for package management. To
run the code, install poetry, run

1. `mkdir assets`
1. `poetry install`
1. `poetry run python main.py`
