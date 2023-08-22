"""
Train and evaluate models.
"""

from dataclasses import dataclass, field
import os
from typing import Optional
import warnings

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn, Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel
from tqdm import tqdm


@dataclass
class TrainerArguments:
    overwrite_output_dir: bool = field(default=False)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    num_train_epochs: int = field(default=1)
    num_workers: int = field(default=0)
    patience: Optional[int] = field(default=None)
    device: str = field(default="cpu")

    def __post_init__(self) -> None:
        if self.per_device_train_batch_size % 64 != 0 or self.per_device_eval_batch_size % 64 != 0:
            warnings.warn("A100s are most efficient for batch sizes that is are multiples of 64.")
        if self.num_workers < 0:
            self.num_workers = int(len(os.sched_getaffinity(0)) / self.num_workers)
        assert self.num_train_epochs > 0
        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")


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
        for _ in tqdm(epochs, leave=True):
            self.model = self.model.to(self.args.device)
            self.model = self.model.train()
            loader = self.get_dataloader(train_dataset, self.args.per_device_train_batch_size)
            train_loss = 0.0
            for batch in tqdm(loader, leave=False):
                labels = batch.pop("labels").to(self.args.device)
                inputs = {k: v.to(self.args.device) for k, v in batch.items()}
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
                inputs = {k: v.to(self.args.device) for k, v in batch.items()}
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
        for l in self.log[-self.args.patience + 1 :]:
            if l["eval_loss"] < self.log[-self.args.patience]["eval_loss"]:
                return False
        return True
