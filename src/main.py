"""
Fine-tune BERT for text classification.
"""

from dataclasses import dataclass, field
from functools import partial
import math
import os
import sys
from typing import Optional
import warnings

from datasets import load_dataset, Dataset, DatasetDict
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PretrainedModel,
    PretrainedTokenizer,
)


BR = "-" * 78


@dataclass
class Arguments:
    dataset_name_or_path: str = field(
        default="ag_news",
        metadata={"help": "Try ``, ``, or ``."}
    )
    pretrained_model_name_or_path: str = field(
        default="bert-large-uncased",
        metadata={"help": "Try ``, ``, or ``."}
    )
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    per_device_batch_size: int = field(default=8)
    num_training_epochs: int = field(default=1)


def count_parameters(model: nn.Module, requires_grad: bool = False) -> int:
    """Counts the parameters in a torch.nn.Module object."""
    return sum(p.numel() for p in model.parameters() if (not requires_grad or p.requires_grad))


# Parse the command-line arguments.
parser = HfArgumentParser([Arguments])
args = parser.parse_args_into_dataclasses()[0]

# Fetch the pretrained fast tokenizer.
tokenizer: PretrainedTokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
def tok_fn(examples, padding: bool, truncation: bool, max_length: Optional[int] = None):
    """Tokenizes text into discrete vectors according to the learned vocabulary."""
    return tokenizer(examples, padding=padding, truncation=truncation, max_length=max_length)


# Fetch and preprocess the dataset.
dataset: DatasetDict | Dataset = load_dataset(args.dataset_name_or_path)
if not isinstance(dataset, DatasetDict):
    dataset = dataset.train_test_split()
dataset = dataset.map(partial(tok_fn, padding=False, truncation=True), batched=True)

model: PretrainedModel = AutoModel.from_pretrained(args.pretrained_model_name_or_path)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


class Trainer:
    """Encapsulates training a Transformer Causal Language Model."""
    def __init__(
        self,
        model: PretrainedModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        data_collator: DataCollatorForLanguageModeling,
    ) -> None:
        ...

    def train() -> None:
        ...

    def eval(self, eval_dataset: Dataset = None) -> dict[str, float]:
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset


trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)

print(f"{tokenizer=}\n{BR}", flush=True)
print(f"{dataset=}\n{BR}", flush=True)
print(f"{data_collator=}\n{BR}", flush=True)
print(f"{config=}\n{BR}", flush=True)
print(f"{model=}\n{BR}", flush=True)
print(f"{count_parameters(model)=}{BR}", flush=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

if training_args.per_device_train_batch_size % 64 != 0:
    warnings.warn("Batch size should be a multiple of 64 for A100s.")

if training_args.do_train:
    trainer.train()

if training_args.do_eval:
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

sys.exit(0)
