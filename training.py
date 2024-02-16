#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt

import yaml

import torch
import pytorch_lightning as pl

from torch.nn.utils import prune
from torchmetrics import MeanSquaredError

from hdnnpy.model import HDNNP
from hdnnpy.data import prepare_dataset
from hdnnpy.lightning_models import LightningHDNNP

t = time()

if torch.cuda.is_available():
    device = torch.device("cuda")
    accelerator = "gpu"
else:
    device = torch.device("cpu")
    accelerator = "cpu"

#######################
# Parse settings file #
#######################
with open('settings.yaml', 'r') as f:
    settings = yaml.load(f, yaml.Loader)

###############
# Seed Things #
###############
if 'seed' in settings:
    pl.seed_everything(settings['seed'])


###################
# Prepare Dataset #
###################
train_dl, val_dl = prepare_dataset(settings['training_data'],
                                   settings['labels_file'],
                                   bs=settings['bs'],
                                   split=settings['train_val_split'],
                                   seed=settings['data_split_seed'],
                                   pin_memory=torch.cuda.is_available())

################
# Create Model #
################
pl_model = LightningHDNNP(dims=settings['dims'], lr=settings['lr'],
                          gamma=settings['gamma'], loss=settings['loss'],
                          optimizer=settings['optimizer'],
                          T_0=settings['T_0'], T_mult=settings['T_mult'],
                          min_lr=settings['min_lr'])


# If we have mask files specified in settings, apply them as pruning
# mask should be a dict mapping element string to a txt file with
# the mask as a column.
# E.g. np.loadtxt(settings['mask'][el]) should give a
# 1D mask for element el
if 'masks' in settings:
    maskfiles = settings['masks']
    masks = {el:torch.tensor(np.loadtxt(maskfile))
            for el,maskfile in maskfiles.items()}
    for el, mask in masks.items():
        width = settings['dims'][el][1]
        # mask is shape (D)
        # weight is shape (M,D)
        # Want to expand mask to (M,D) with expanded_mask[i,d] = mask[d]
        mask = torch.tile(mask, [width, 1])
        # Apply the mask to prune the model
        prune.custom_from_mask(pl_model.model.networks[el].layers[0],
                               'weight',
                               mask)


################################
# Create and Configure Trainer #
################################
# Custom CSV Logs
logger_csv = pl.loggers.csv_logs.CSVLogger('logs-csv')
#logger_tb = pl.loggers.tensorboard.TensorBoardLogger('logs-tb')
#logger = [logger_csv, logger_tb]  # Use Both CSV and TensorBoard Loggers
logger = logger_csv

lr_monitor = pl.callbacks.LearningRateMonitor()
callbacks = [lr_monitor]

# Checkpointing
checkpointer = pl.callbacks.ModelCheckpoint(monitor='val_rmse', save_last=True,
                                       save_top_k=5,
                                       filename='{epoch}-{val_rmse:.8f}')
callbacks.append(checkpointer)

amp_plugin = pl.plugins.precision.NativeMixedPrecisionPlugin(16, "cuda")  # For older versions of pl
trainer = pl.Trainer(accelerator=accelerator, logger=logger,
                     max_epochs=settings['epochs'],
                     enable_progress_bar=False,
                     callbacks=callbacks,
                     track_grad_norm=2,
                     max_time=settings['max_time'],
                     plugins=[amp_plugin])

if not 'continue_checkpt' in settings:
    settings['continue_checkpt'] = False
if settings['continue_checkpt']:
    print(f'Continuing from Checkpoint at: {settings["checkpt_path"]}')
    trainer.fit(pl_model, train_dl, val_dl, ckpt_path=settings['checkpt_path'])
else:
    print('Training Model')
    trainer.fit(pl_model, train_dl, val_dl)
