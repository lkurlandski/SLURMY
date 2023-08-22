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
from typing import Any

sys.path.insert(0, ".")

from datasets import load_dataset, Dataset, DatasetDict
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.helpers import OutputHelper
from src.trainer import Trainer, TrainerArguments
from src.utils import count_parameters, pformat_log


BR = "-" * 78
VALID = {
    "datasets": [
        "ag_news",
        "imdb",
        "sst2",
        "trec",
    ],
    "models": [
        "prajjwal1/bert-tiny",
        "prajjwal1/bert-mini",
        "prajjwal1/bert-medium",
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "roberta-large",
    ],
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


def tokenize(examples, tokenizer: PreTrainedTokenizer, **kwds) -> Any:
    """Tokenizes text into discrete vectors according to the learned vocabulary."""
    return tokenizer(examples["text"], **kwds)


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
    if args.dataset_name == "trec":
        dataset = dataset.remove_columns(["coarse_label"])
        dataset = dataset.rename_columns({"fine_label": "label"})

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
    main(args, trainer_args)
    print(f"Finishing @{datetime.now()}")
    print(BR, flush=True)

    sys.exit(0)


if __name__ == "__main__":
    cli()
