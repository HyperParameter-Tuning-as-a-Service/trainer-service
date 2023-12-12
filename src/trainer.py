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

class CustomTrainer():
    '''
    Trainer class for loading the trainer class in memory and running training, evaluation, and storing
    '''
    def __init__(self, repo_name, hyperparameters_path, tokenizer, model, dataset) -> None:
        self.repo_name = repo_name
        f = open(hyperparameters_path)
        self.hyper = json.load(f)
        self.model = model
        self.rouge = evaluate.load("rouge")

        self.training_args = TrainingArguments(
            output_dir=self.repo_name,
            learning_rate=self.hyper["learning_rate"],
            per_device_train_batch_size=self.hyper["batch_size"],
            per_device_eval_batch_size=self.hyper["batch_size"],
            num_train_epochs=self.hyper["max_epochs"],
            weight_decay=self.hyper["weight_decay"],
            warmup_steps=self.hyper["warmup_steps"],
            save_strategy="epoch",
            evaluation_strategy="epoch"
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset.tokenized_train,
            eval_dataset=dataset.tokenized_test,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator
        )

        print("Loaded trainer")

    def compute_metrics(self, eval_pred):
        load_accuracy = load_metric("accuracy")
        load_f1 = load_metric("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}

    def compute_metrics_s(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

    def train(self):
        train_events = self.trainer.train()
        print(train_events)

    def evaluate(self):
        eval_events = self.trainer.evaluate()
        print(eval_events)

    def get_model(self):
        return self.model

    def save(self, save_path, user_name):
        self.model.save_pretrained(save_path + f"customer_model_{user_name}", from_pt=True)
        print("model saved")