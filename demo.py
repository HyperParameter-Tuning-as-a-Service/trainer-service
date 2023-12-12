from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, DefaultDataCollator
from transformers import AutoModelForMaskedLM, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import load_dataset, load_metric
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import ast
import evaluate
import sys

from src.datasets import LoadDataset
from src.models import LoadModel
from src.trainer import CustomTrainer

def main():
    device = "cuda:0"
    task_type="summarization"
    repo_name = "finetuning-s-model-3000-samples"
    hyperparams = "./data/hyperparameters.json"
    train_path="./data/train_summarization.csv"
    test_path="./data/test_summarization.csv"
    tokenizer_name = "stevhliu/my_awesome_billsum_model"
    model_name = "stevhliu/my_awesome_billsum_model"
    save_path = "./output"
    user_name = "s1"

    model_c = LoadModel(device=device,
                        tokenizer_name=tokenizer_name,
                        model_name=model_name,
                        task_type=task_type)

    dataset = LoadDataset(train_path=train_path,
                          test_path=test_path,
                          model=model_c,
                          task_type=task_type)

    tokenizer, model = model_c.get_model()

    trainer = CustomTrainer(repo_name=repo_name,
                            hyperparameters_path=hyperparams,
                            tokenizer=tokenizer,
                            model=model,
                            dataset=dataset)

    trainer.train()
    trainer.evaluate()
    trainer.save(save_path=save_path,
                 user_name=user_name)

if __name__ == "__main__":
    main()