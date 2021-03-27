import os
import json
import logging
import random
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from iqradre.extract.data import utils 
from iqradre.extract.data import loader
from iqradre.extract.data.dataset import IDCardDataset
from iqradre.extract.config import label as label_cfg
from iqradre.extract.trainer.task import TaskLayoutLM


from transformers import BertTokenizer, AutoModel
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    LayoutLMConfig,
    LayoutLMForTokenClassification,
    get_linear_schedule_with_warmup,
)

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


SAVED_CHECKPOINT_PATH = "checkpoints/v2/"
SAVED_LOGS_PATH = "logs/v2/"

SUMMARY = "top"
MANUAL_SEED = 1261
MAX_EPOCH = 2
MAX_STEPS = None
VALCHECK_INTERVAL = 2000
NUM_GPUS = 1
DISTRIBUTED_BACKEND = None
LOG_FREQ = 100
DETERMINISTIC = True
BENCHMARK = True
CHECKPOINT_PATH = None
BATCH_SIZE = 4
NUM_WORKERS = 16

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


tokenizer = BertTokenizer.from_pretrained(
    "indobenchmark/indobert-base-p2",
    do_lower_case=True,
    cache_dir=None,
)

path = '/data/idcard/results/combined/layoutlm/20kv2/1616353649'
train_loader, valid_loader = loader.get_loader(
    path, tokenizer=tokenizer, 
    batch_size=BATCH_SIZE, 
    num_workers=NUM_WORKERS,
    rand_seq=True,
    rand_seq_prob=0.5,
    excluce_image_mask=True
)

config = LayoutLMConfig.from_pretrained(
    "microsoft/layoutlm-base-uncased",
    num_labels=label_cfg.num_labels,
    cache_dir=None
)

model = LayoutLMForTokenClassification.from_pretrained(
    'microsoft/layoutlm-base-uncased',
    config=config,
#     return_dict=True
)

model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

state_dict = torch.load('weights/extract/layoutlm_v2_ktp_20kv1_vacc_0.981_vloss_0.26.pth', map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model = model.to(device)


task = TaskLayoutLM(model, tokenizer)


# DEFAULTS used by the Trainer
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=SAVED_CHECKPOINT_PATH,
#     save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix='layoutlm-v2'
)

tb_logger = pl_loggers.TensorBoardLogger(SAVED_LOGS_PATH)
pl.trainer.seed_everything(MANUAL_SEED)


trainer = pl.Trainer(
    weights_summary=SUMMARY,
    max_epochs=MAX_EPOCH,
    max_steps=MAX_STEPS,
#     val_check_interval=1000,
    gpus=1,
    distributed_backend=DISTRIBUTED_BACKEND,
    log_every_n_steps=LOG_FREQ,
    deterministic=DETERMINISTIC,
    benchmark=BENCHMARK,
    logger=tb_logger, 
    checkpoint_callback=checkpoint_callback, 
#     resume_from_checkpoint=CHECKPOINT_PATH
)

trainer.fit(task, train_loader, valid_loader)

torch.save(model.state_dict(), 'weights/extract/layoutlm_v2_ktp_20kv1_vacc_0.981_vloss_0.26_train_cli.pth')
