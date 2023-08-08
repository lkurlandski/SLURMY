"""
Pretrain GPT-2 on Causal Language Modeling (CLM) task.

Based on the guide from:
    https://huggingface.co/docs/transformers/v4.31.0/en/tasks/language_modeling

GPT2 Paper:
    https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
"""

from dataclasses import dataclass, field
import math
import os
import sys
import warnings

from datasets import load_dataset
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


DATASET_NAME_OR_PATH = "eli5"
SUBSET = 16384
SPLIT = f"train_asks[0:{SUBSET}]"
BLOCK_SIZE = 128
NUM_PROC = 4
BR = "-" * 78


@dataclass
class Arguments:
    test_size: float = field(
        default=0.1, metadata={"help": "Are you stupid? This is obviously the test size."},
    )
    pretrained_model_name_or_path: str = field(
        default="gpt2", metadata={"help": "Try `gpt2`, `gpt_neo`, `gpt_neox` etc."},
    )
    scale: float = field(
        default=1.0, metadata={"help": "Up/down scale gpt2 model by this factor."},
    )


def count_parameters(model: nn.Module, requires_grad: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (not requires_grad or p.requires_grad))


parser = HfArgumentParser([Arguments, TrainingArguments])
args, training_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token


def tok_fn(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])


def grp_fn(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= BLOCK_SIZE:
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


dataset = load_dataset(DATASET_NAME_OR_PATH, split=SPLIT)
dataset = dataset.train_test_split(test_size=args.test_size)
dataset = dataset.flatten()
dataset = dataset.map(tok_fn, batched=True, num_proc=NUM_PROC, remove_columns=dataset["train"].column_names)
dataset = dataset.map(grp_fn, batched=True, num_proc=NUM_PROC)

if args.pretrained_model_name_or_path == "gpt2" and args.scale != 1:
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
else:
    if args.scale != 1:
        warnings.warn(
            f"Ignoring `{args.scale=}` as it is not implemented for models other than `gpt2`.",
            flush=True,
        )
    config = AutoConfig.from_pretrained(args.pretrained_model_name_or_path)
    model = AutoModelForCausalLM.from_config(config)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
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
