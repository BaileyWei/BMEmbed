import os
import json
import logging
import random
from pathlib import Path
from typing import List, Optional
from typing import Any, Dict, Tuple, Union

import datasets
import torch
import transformers
from dataclasses import dataclass, field
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModel, Trainer
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model
from loss import HardNegativeNLLLoss, ListwiseLoss, ListMLELoss
from models import QwenForSequenceEmbedding
logger = get_logger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The PEFT model checkpoint to add on top of base model.")},
    )
    max_seq_length: Optional[int] = field(
        # default=None,
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        # default=None,
        default='bfloat16',
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5"},
    )
    dataset_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file or folder."}
    )
    # TODO: implement this
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    simcse_dropout: float = field(
        default=0.1, metadata={"help": "The SimCSE dropout rate for the model"}
    )

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    cache_dir: Optional[str] = field(
        default='./',
        metadata={"help": "huggingface cache dir"}
    )

    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={
            "help": "The loss class to use for training. Options: HardNegativeNLLLoss, MaxLikelihoodLoss, CrossEntropyLoss"
        },
    )

    listwise: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use listwise training strategy"
            )
        },
    )

    label_scaling: float = field(
        default=0.5,
        metadata={"help": "The loss scale for the loss function"}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: Optional[str] = field(
        default='../output/',
        metadata={
            "help": (
                " "
                "value if set."
            )
        },
    )

    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )

    warmup_steps: int = field(default=100, metadata={"help": "Linear warmup over warmup_steps."})

    logging_steps: float = field(
        default=1,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    n_gpu: int = field(init=False, repr=False, default=1)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    (model_args,
     data_args,
     training_args,
     custom_args) = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(kwargs_handlers=[])
    cache_dir = custom_args.cache_dir
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(training_args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              cache_dir=cache_dir,
                                              add_eos_token=True)
    if not tokenizer.pad_token:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    dataset = {'train': json.load(open(f'{data_args.dataset_file_path}/bm25_dataset.json'))}
    # dataset = {'train': json.load(open(f'{data_args.dataset_file_path}'))}

    # Log a few random samples from the training set:
    for index in random.sample(range(len(dataset["train"])), 1):
        logger.info(f"Sample {index} of the training set: {dataset['train'][index]}.")
    # base model
    model = QwenForSequenceEmbedding(model_args.model_name_or_path, cache_dir=cache_dir)
    # model.print_trainable_parameters()

    # import ipdb;ipdb.set_trace()
    accelerator.print(model)

    def collate_fn_multi_negatives(batch):
        instruction = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: '

        if isinstance(batch[0]['positive'], dict):
            sentences = [f"{instruction}{item['query']}" for item in batch]
            positives = [item['positive']['text'] for item in batch]
            negatives = [n['text'] for item in batch for n in item['negative']]
            labels = [[item['positive']['score']] + [negative['score'] for negative in item['negative']] for item in
                      batch]

        elif isinstance(batch[0]['positive'], str):
            sentences = [f"{instruction}{item['query']}" for item in batch]
            positives = [item['positive'] for item in batch]
            negatives = [n for item in batch for n in item['negative']]
            labels = [0]

        result = []

        sentence_batch_dict = tokenizer(sentences, max_length=model_args.max_seq_length, return_attention_mask=True,
                                        padding=True, truncation=True, return_tensors='pt')
        positives_batch_dict = tokenizer(positives, max_length=model_args.max_seq_length, return_attention_mask=True,
                                         padding=True, truncation=True, return_tensors='pt')
        negatives_batch_dict = tokenizer(negatives, max_length=model_args.max_seq_length, return_attention_mask=True,
                                         padding=True, truncation=True, return_tensors='pt')

        result.append(sentence_batch_dict)
        result.append(positives_batch_dict)
        result.append(negatives_batch_dict)

        return result, labels

    def collate_fn(batch):

        instruction = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: '

        sentences = [f"{instruction}{item['query']}"  for item in batch]

        negatives = [item['negative'] for item in batch]
        positives = [item['positive'] for item in batch]

        result = []
        sentence_batch_dict = tokenizer(sentences, max_length=model_args.max_seq_length, return_attention_mask=True,
                                        padding=True, truncation=True, return_tensors='pt')
        positives_batch_dict = tokenizer(positives, max_length=model_args.max_seq_length, return_attention_mask=True,
                                         padding=True, truncation=True, return_tensors='pt')
        negatives_batch_dict = tokenizer(negatives, max_length=model_args.max_seq_length, return_attention_mask=True,
                                         padding=True, truncation=True, return_tensors='pt')

        result.append(sentence_batch_dict)
        result.append(positives_batch_dict)
        result.append(negatives_batch_dict)

        labels = [0]
        return result, labels

    class MySupervisedTrainer(Trainer):
        def __init__(
                self,
                *args,
                listwise=None,
                loss_function=None,
                output_dir=None,
                **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.loss_function = loss_function
            self.listwise = listwise
            self.train_dataloader = DataLoader(
                dataset['train'],
                shuffle=True,
                collate_fn=collate_fn_multi_negatives if self.listwise else collate_fn,
                batch_size=self._train_batch_size,
                pin_memory=True,
            )
            self.output_dir = output_dir

        def get_train_dataloader(self):
            return self.accelerator.prepare(self.train_dataloader)

        def compute_loss(
                self,
                model: nn.Module,
                inputs: Dict[str, Union[torch.Tensor, Any]],
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
            features, labels = inputs
            q_reps = self.model(**features[0])
            d_reps = self.model(**features[1])
            d_reps_neg = self.model(**features[2])

            # scores = self.similarity_fct(q_reps, d_reps_neg) * self.scale
            # # Example a[i] should match with b[i]
            # range_labels = torch.arange(0, scores.size(0), device=scores.device)
            # logger.info(d_reps.shape, d_reps_neg.shape)
            if self.listwise:
                labels = torch.tensor(labels, device=d_reps_neg.device)
                loss = self.loss_function(q_reps, d_reps, d_reps_neg,
                                          labels=labels
                                          )
            else:
                loss = self.loss_function(q_reps, d_reps, d_reps_neg,
                                          )

            return loss

        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            checkpoint = output_dir.strip('/').split('/')[-1]

            os.makedirs(f'{self.output_dir}/{checkpoint}', exist_ok=True)
            logger.info(f"Saving model checkpoint to {self.output_dir}/{checkpoint}")
            self.model.model.save_pretrained(f'{self.output_dir}/{checkpoint}')
            self.tokenizer.save_pretrained(f'{self.output_dir}/{checkpoint}')



    if custom_args.listwise and custom_args.loss_class == 'MaxLikelihoodLoss':
        loss_fn = ListMLELoss(label_scaling=custom_args.label_scaling)
    elif custom_args.listwise and custom_args.loss_class == 'CrossEntropyLoss':
        loss_fn = ListwiseLoss(label_scaling=custom_args.label_scaling)
    else:
        loss_fn = HardNegativeNLLLoss()

    name_data_file = data_args.dataset_file_path.strip('/').split('/')[-1]
    method = custom_args.loss_class
    output_dir = f'{training_args.output_dir}/{data_args.dataset_name}/{method}/qwen/{name_data_file}_{custom_args.label_scaling}'


    trainer = MySupervisedTrainer(
        model=model,
        args=training_args,
        listwise=custom_args.listwise,
        tokenizer=tokenizer,
        loss_function=loss_fn,
        output_dir=output_dir
    )

    trainer.train()
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
