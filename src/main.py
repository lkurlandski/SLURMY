"""
Fine-tune BERT for text classification.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from functools import partial
import os
from pathlib import Path
from pprint import pformat
import sys
from typing import Any, ClassVar, Optional
import warnings

from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn, Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm


BR = "-" * 78
VALID = {
    "datasets": [
        "ag_news",
    ],
    "models": [
        "prajjwal1/bert-tiny",
        "prajjwal1/bert-mini",
        "prajjwal1/bert-medium",
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "roberta-large",
    ]
}


@dataclass
class Arguments:
    dataset_name: str = field(
        default="ag_news", metadata={"help": f"Try one of {pformat(VALID['datasets'])}."}
    )
    model_name: str = field(
        default="bert-tiny", metadata={"help": f"Try one of {pformat(VALID['models'])}."}
    )
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    do_anal: bool = field(default=False)
    max_length: int = field(default=512)
    cleanup_cache_files: bool = field(default=False)
    output_root: Path = field(default="./output")

    def __post_init__(self) -> None:
        assert self.dataset_name in VALID["datasets"]
        assert self.model_name in VALID["models"]
        assert 0 < self.max_length <= 512


@dataclass
class TrainerArguments:
    overwrite_output_dir: bool = field(default=False)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    num_train_epochs: int = field(default=1)
    num_workers: int = field(default=0)
    patience: int = field(default=5)
    device: str = field(default="cpu")

    def __post_init__(self) -> None:
        if self.per_device_train_batch_size % 64 != 0 or self.per_device_eval_batch_size % 64 != 0:
            warnings.warn("A100s are most efficient for batch sizes that is are multiples of 64.")
        if self.num_workers < 0:
            self.num_workers = int(len(os.sched_getaffinity(0)) / self.num_workers)
        assert self.num_train_epochs > 0
        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")


class OutputHelper:

    _model_path: ClassVar[str] = "model/"
    _trainer_log_file: ClassVar[str] = "trainer_log.json"
    _test_results_file: ClassVar[str] = "test_results.json"
    _learning_curve_file: ClassVar[str] = "learning_curve.png"

    def __init__(self, output_root: Path, dataset_name: str, model_name: str) -> None:
        self.output_root = output_root
        self.dataset = dataset_name
        self.model_name = model_name

    @property
    def path(self) -> Path:
        return self.output_root / self.dataset / self.model_name

    @property
    def trainer_log_file(self) -> Path:
        return self.path / self._trainer_log_file

    @property
    def test_results_file(self) -> Path:
        return self.path / self._test_results_file

    @property
    def learning_curve_file(self) -> Path:
        return self.path / self._learning_curve_file

    @property
    def model_path(self) -> Path:
        return self.path / self._model_path

    def mkdir(self, exist_ok: bool = False) -> None:
        self.output_root.mkdir(parents=True, exist_ok=exist_ok)
        self.path.mkdir(parents=True, exist_ok=exist_ok)
        self.model_path.mkdir(parents=True, exist_ok=exist_ok)


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainerArguments,
        data_collator: DataCollatorWithPadding,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else AdamW(model.parameters())
        self.log: list[int, dict[str, float]] = []

    def train(self, train_dataset: Dataset, eval_dataset: Dataset) -> list[dict[str, float]]:
        epochs = list(range(1, self.args.num_train_epochs + 1))
        for epoch in tqdm(epochs, leave=True):
            self.model = self.model.to(self.args.device)
            self.model = self.model.train()
            loader = self.get_dataloader(train_dataset, self.args.per_device_train_batch_size)
            train_loss = 0.0
            for batch in tqdm(loader, leave=False):
                labels = batch.pop("labels").to(self.args.device)
                inputs = {k : v.to(self.args.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss: Tensor = self.loss_fn(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss += loss.item()
            log = {"train_loss": train_loss / len(loader)} | self.evaluate(eval_dataset)
            self.log.append(log)
            if self.stop_training():
                break
        return self.log

    def evaluate(self, eval_dataset: Dataset) -> dict[str, float]:
        self.model = self.model.to(self.args.device)
        self.model = self.model.eval()
        loader = self.get_dataloader(eval_dataset, self.args.per_device_eval_batch_size)
        eval_loss = 0.0
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch in tqdm(loader, leave=False):
                labels = batch.pop("labels").to(self.args.device)
                inputs = {k : v.to(self.args.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                loss: Tensor = self.loss_fn(outputs.logits, labels)
                eval_loss += loss.item()
                y_pred.extend(torch.argmax(outputs.logits, dim=1).detach().cpu().tolist())
                y_true.extend(labels.detach().cpu().tolist())
        return {
            "eval_loss": eval_loss / len(loader),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1-score": f1_score(y_true, y_pred, average="macro"),
        }

    def get_dataloader(self, dataset: Dataset, batch_size: int = 1) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

    def stop_training(self) -> bool:
        if self.args.patience is None:
            return False
        if len(self.log) < self.args.patience:
            return False
        for l in self.log[-self.args.patience + 1:]:
            if l["eval_loss"] < self.log[-self.args.patience]["eval_loss"]:
                return False
        return True   


def count_parameters(model: nn.Module, requires_grad: bool = False) -> int:
    """Counts the parameters in a torch.nn.Module object."""
    return sum(p.numel() for p in model.parameters() if (not requires_grad or p.requires_grad))


def tokenize(examples, tokenizer: PreTrainedTokenizer, **kwds) -> Any:
    """Tokenizes text into discrete vectors according to the learned vocabulary."""
    return tokenizer(examples["text"], **kwds)


def pformat_log(log: list[dict[str, float]]) -> dict[str, list[float]]:
    """Pretty formats the log for plotting."""
    new_log = {k: [] for k in log[0].keys()}
    for l in log:
        for k, v in l.items():
            new_log[k].append(v)        
    return new_log


def main(args: Arguments, trainer_args: TrainerArguments) -> None:
    """Main function."""
    # Fetch the tokenizer and dataset. 
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset: DatasetDict | Dataset = load_dataset(args.dataset_name)

    print(f"{tokenizer=}")
    print(f"{dataset=}")
    print(f"{BR}", flush=True)

    # Peprocess the dataset.
    tokenize_fn = partial(
        tokenize,
        tokenizer=tokenizer,
        padding=False,  # We will dynamically pad the data during training,
        truncation=True,  # but we will truncate the data to BERT's max length
        max_length=args.max_length,  # which is a hyperparameter common to all the BERT models.
        return_attention_mask=False,  # Attention mask needs to be created when sequences are padded
    )
    dataset = dataset.map(tokenize_fn, batched=True, num_proc=1)
    dataset = dataset.remove_columns(["text"])

    print(f"{tokenize_fn=}")
    print(f"{dataset=}")
    print(f"{BR}", flush=True)

    # Create train/validation/test splits.
    if not isinstance(dataset, DatasetDict):
        dataset = dataset.train_test_split()
    if not isinstance(dataset["train"], DatasetDict):
        dataset["train"] = dataset["train"].train_test_split()
    if args.cleanup_cache_files:
        dataset.cleanup_cache_files()

    print(f"{dataset=}")
    print(f"{BR}", flush=True)

    # Exit early if we are not training or evaluating.
    if not any((args.do_train, args.do_eval, args.do_anal)):
        return

    # Set up the model for training.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=dataset["train"]["train"].features["label"].num_classes,
    )
    trainer = Trainer(
        model=model,
        args=trainer_args,
        data_collator=data_collator,
    )

    print(f"{data_collator=}")
    print(f"{model=}")
    print(f"{count_parameters(model)=}")
    print(BR, flush=True)

    # Disable some annoying warnings.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    oh = OutputHelper(args.output_root, args.dataset_name, args.model_name)
    oh.mkdir(exist_ok=trainer_args.overwrite_output_dir)

    if args.do_train:
        log = trainer.train(dataset["train"]["train"], dataset["train"]["test"])
        model.save_pretrained(oh.model_path)
        with open(oh.trainer_log_file, "w") as fp:
            json.dump(log, fp, indent=2)

    if args.do_eval:
        results = trainer.evaluate(dataset["test"])
        with open(oh.test_results_file, "w") as fp:
            json.dump(results, fp, indent=2)

    if args.do_anal:
        with open(oh.trainer_log_file, "r") as fp:
            log = json.load(fp)
        log = pformat_log(log)
        print(pformat(log))
        epochs = list(range(1, len(log["train_loss"]) + 1))
        plt.plot(epochs, log["eval_loss"], label="loss", color="blue")
        plt.plot(epochs, log["f1-score"], label="f1-score", color="green")
        plt.plot(epochs, log["accuracy"], label="accuracy", color="red")
        plt.legend()
        plt.xlim(0, len(log) + 1)
        plt.ylim(0, 1.0)
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.grid()
        plt.savefig(oh.learning_curve_file)


def cli() -> None:
    """Parse the command-line arguments and run main function."""
    parser = HfArgumentParser([Arguments, TrainerArguments])
    args, trainer_args = parser.parse_args_into_dataclasses()
    print(f"{args=}")
    print(f"{trainer_args=}")
    print(BR, flush=True)

    print(f"Starting @{datetime.now()}")
    print(BR, flush=True)
    _ = main(args, trainer_args)
    print(f"Finishing @{datetime.now()}")
    print(BR, flush=True)

    sys.exit(0)


if __name__ == "__main__":
    cli()
