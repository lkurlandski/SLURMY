"""
Pretrain GPT-2 on Causal Language Modeling (CLM) task.

Based on the guide from:
    https://huggingface.co/docs/transformers/v4.31.0/en/tasks/language_modeling

GPT2 Paper:
    https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
"""

from dataclasses import dataclass, field
from functools import partial
import math
import os
import sys
from typing import Optional
import warnings

from datasets import load_dataset, Dataset
from torch import nn
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    HfArgumentParser,
    PretrainedModel,
    Trainer,
    TrainingArguments,
)


DATASET_NAME_OR_PATH = "eli5"
PRETRAINED_MODEL_NAME_OR_PATH = "gpt2"
NUM_PROC = 4
BR = "-" * 78


@dataclass
class Arguments:
    test_size: float | int = field(default=0.1, metadata={"help": "The test size."},)
    scale: float = field(default=1.0, metadata={"help": "Scale gpt2 model by this factor."})
    subset: Optional[int] = field(default=None, metadata={"help": "Use a subset of the dataset."})
    block_size: int = field(default=128, metadata={"help": "Size of language modeling chunks."})


def count_parameters(model: nn.Module, requires_grad: bool = False) -> int:
    """Counts the parameters in a torch.nn.Module object."""
    return sum(p.numel() for p in model.parameters() if (not requires_grad or p.requires_grad))


# Parse the command-line arguments.
parser = HfArgumentParser([Arguments, TrainingArguments])
args, training_args = parser.parse_args_into_dataclasses()

# Fetch the pretrained fast tokenizer.
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token


def tok_fn(examples):
    """Tokenizes text into discrete vectors according to the learned vocabulary."""
    return tokenizer([" ".join(x) for x in examples["answers.text"]])


def grp_fn(examples, block_size: int):
    """Groups several short examples together into a single sample."""
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Fetch and preprocess the dataset.
split = "train_asks"
if args.subset:
    split += f"[0:{args.subset}]"
dataset = load_dataset(DATASET_NAME_OR_PATH, split=split)
dataset = dataset.train_test_split(test_size=args.test_size)
dataset = dataset.flatten()
dataset = dataset.remove_columns(dataset["train"].column_names)
dataset = dataset.map(tok_fn, batched=True, num_proc=NUM_PROC)
dataset = dataset.map(partial(grp_fn, args.block_size), batched=True, num_proc=NUM_PROC)

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=int(1024 * args.scale),
    n_embd=int(768 * args.scale),
    n_layer=int(12 * args.scale),
    n_head=int(12 * args.scale),
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


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
    args=training_args,
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
