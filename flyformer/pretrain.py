#!/usr/bin/env python
# coding: utf-8

# TRANSFOMERS VERSION 4.6.0

"""
Flyformer pretrainer.

Run using docker-compose via `jorgebotas/flyformer-docker``
Environment variables defined in Dockerfile

ROOT_DIR: project directory. Should mount input dir. See `docker-compose.yaml`
DATA_DIR: pretraining data directory. Includes .dataset (see
          `flyformer.data_preprocessing` and `flyformer.tokenizer`
          for more information).

"""

# ----- IMPORTS ----------------------------------------------------------------
from collections import namedtuple
import datetime
import os
from pathlib import Path
import pickle
import random
from typing import Union

from datasets import load_from_disk
import numpy as np
import pytz
import tensorrt
import torch
from transformers import BertConfig, BertForMaskedLM, TrainingArguments, logging

from geneformer import GeneformerPretrainer

print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

# ----- HELPER FUNCTIONS -------------------------------------------------------
def path(*args: str) -> Path:
    """Return relative path to root directory"""
    return Path(os.path.join(ROOT_DIR, *args))

def get_run_name(timezone: str = "US/Eastern") -> str:
    """Return run name with datestamp, model and training parameters"""

    def get_model(key: str) -> Union[str, float]:
        """Return value of key in MODEL_PARAMETERS"""
        return MODEL_PARAMETERS.get(key)

    def get_training(key: str) -> Union[str, float]:
        """Return value of key in TRAINING_PARAMETERS"""
        return TRAINING_PARAMETERS.get(key)

    # set local time/directories
    timezone = pytz.timezone(timezone)
    date = datetime.datetime.now(tz=timezone)
    datestamp = f"{str(date.year)[-2:]}.{date.month:02d}.{date.day:02d}_"
    datestamp += f"{date.strftime('%X').replace(':','.')}"
    run = f"{datestamp}__"
    run += f"NL-{get_model('num_hidden_layers')}__"
    run += f"EMB-{get_model('hidden_size')}__"
    run += f"ID-{get_model('max_position_embeddings')}__"
    run += f"E-{get_training('num_train_epochs')}__"
    run += f"B-{get_training('per_device_train_batch_size')}__"
    run += f"LR-{get_training('learning_rate')}__"
    run += f"LS-{get_training('lr_scheduler_type')}__"
    run += f"WU-{get_training('warmup_steps')}"
    return run

def read_pickle(path: Path) -> dict:
    """Read .pickle file from path"""
    with open(path, "rb") as fp:
        return pickle.load(fp)


# ----- GLOBAL VARIABLES -------------------------------------------------------
# Defined in Dockerfile

# Directories
ROOT_DIR = Path(os.environ["ROOT_DIR"])
# Directory containing .dataset (Apache Arrow format), example lengths and
# token dictionary
DATA_DIR = ROOT_DIR / "input"

# Seeds
RANDOM_SEED = int(os.environ["RANDOM_SEED"])
TORCH_SEED = int(os.environ["TORCH_SEED"])
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)

# Set logging preferences
logging.enable_default_handler()
logging.set_verbosity(logging.DEBUG)

# Load token dictionary
TOKEN_DICT = read_pickle(DATA_DIR / "token_dictionary.pickle")


# ----- LOAD DATASET -----------------------------------------------------------
print("> Loading dataset...")
dataset = load_from_disk(next(DATA_DIR.glob("*.dataset"))
print(f"> {dataset.num_rows} examples loaded")


# ----- MODEL PARAMETERS -------------------------------------------------------
# Define (BERT) model parameters
n_embedding_dimensions = 256
MODEL_PARAMETERS = {
    # Transformer
    "model_type": "bert",
    # Number of hidden layers (transformer encoder units)
    # Each composed of self-attention and feed forward layers
    "num_hidden_layers": 6,
    # Number of attention heads per hidden layer (4)
    "num_attention_heads": 4,
    # Vocabulary size of the model = #genes + 2 (<mask> and <pad> tokens)
    "vocab_size": len(TOKEN_DICT),
    # <pad> token id from TOKEN_DICT
    "pad_token_id": TOKEN_DICT.get("<pad>"), # BERT default = 0 (idem)
    # Max input size, i.e. max sequence length (2^11 = 2048)
    "max_position_embeddings": 2**11,
    # Dimensionality of the encoder layers and the pooler layer
    "hidden_size": n_embedding_dimensions,
    # Dimensionality of the “intermediate” (i.e., feed-forward) layer
    "intermediate_size": n_embedding_dimensions * 2,
    # Non-linear activation function in the encoder and pooler
    # Geneformer used Relu, but should try Gelu (avoids "dead neurons")
    "hidden_act": "relu", # BERT default = "gelu"
    # The dropout probabilitiy for all fully connected layers in the embeddings,
    # encoder, and pooler
    "hidden_dropout_prob": 0.02, # BERT default = 0.1
    # The dropout ratio for the attention probabilities
    "attention_probs_dropout_prob": 0.02, # BERT default = 0.1
    # The standard deviation of the truncated_normal_initializer
    # for initializing all weight matrices
    "initializer_range": 0.02, # BERT default 0.02
    # The epsilon used by the layer normalization layers
    "layer_norm_eps": 1e-12, # BERT default 1e-12
    # Use gradient checkpointing to save memory at the expense of slower
    # backward pass
    "gradient_checkpointing": False, # BERT default = False
}

# ----- TRAINING PARAMETERS ----------------------------------------------------
batch_size = 12
n_saves_epoch = 8 # Number of saves per epoch
TRAINING_PARAMETERS = {
    # Initial learning rate for Adam optimizer
    "learning_rate": 1e-3, # Default 5e-5
    # Whether to run training
    "do_train": True,
    # Whether to run evaluation on dev set
    "do_eval": False,
    # Learning rate scheduler type
    "lr_scheduler_type": "linear", # BERT default = "linear"
    # Number of steps used for a linear warmup from 0 to learning_rate
    "warmup_steps": 10_000,
    # Weight decay applied (if not zero) to all layers except all bias and
    # LayerNorm weights in AdamW optimizer.
    "weight_decay": 1e-3, # BERT default = 0
    # The batch size per GPU/TPU/MPS/NPU core/CPU for training
    "per_device_train_batch_size": batch_size, # BERT default = 8
    # Total number of training epochs to perform
    "num_train_epochs": 3, # BERT default = 3
    # The checkpoint save strategy to adopt during training
    "save_strategy": "steps",
    # Number of updates steps before two checkpoint saves
    # n_saves_epoch (8) saves per epoch
    "save_steps": np.floor((dataset.num_rows / batch_size) / n_saves_epoch),
    # Number of update steps between two logs
    "logging_steps": 10, # 1000
    # Group together samples of roughly the same length in the training dataset
    # (to minimize padding applied and be more efficient). Dynamic padding
    "group_by_length": True, # BERT default = False
    "length_column_name": "length", # BERT default = "length"
    # Enable tqdm progress bar
    "disable_tqdm": False,
}


# ----- CREATE OUTPUT DIRECTORIES FOR MODEL ------------------------------------
# Get run name with datestamp, model and training parameters
run = get_run_name()
# TensorBoard log directory
logging_dir = path(f"output/runs/{run}")
training_dir = path(f"output/models/{run}")
model_dir = path(training_dir, "models")
model_file = path(model_dir, "pytorch_model.bin")
print(f"> Run name: {run}")
# Avoid overwritting previously saved model
if os.path.isfile(model_file):
    raise Exception("Model already saved to this directory.")
# Create training and model directories (mounted to host disk)
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# ----- INITIALIZE MODEL AND START TRAINING ------------------------------------
# Build BERT model
model = BertForMaskedLM(BertConfig(**MODEL_PARAMETERS))
model.train()
# Build Geneformer pretrainer
pretrainer = GeneformerPretrainer(
    model=model,
    args=TrainingArguments(output_dir=training_dir,
                           logging_dir=logging_dir,
                           **TRAINING_PARAMETERS),
    train_dataset=dataset,
    example_lengths_file=next(DATA_DIR.glob("*lengths.pickle")),
    token_dictionary=TOKEN_DICT,
)
# Pretraining
pretrainer.train()
# Save model to disk
pretrainer.save_model(model_dir)
