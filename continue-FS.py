#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:06:58 2023

@author: sandberj
"""

####################################################
# Take a Feature Selection path and for each model #
# continue the training without penalty, and with  #
# the discarded features removed.                  #
####################################################
import sys
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import yaml

import torch
from torch.nn.utils import prune
import pytorch_lightning as pl

from torchmetrics import MeanSquaredError

from hdnnpy.model import HDNNP
from hdnnpy.data import prepare_dataset
from hdnnpy.lightning_models import LightningHDNNP_GL, LightningHDNNP

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
                                   pin_memory=torch.cuda.is_available(),
                                   scaling_file=settings['scaling_file'],
                                   train_workers=settings['train_workers'],
                                   val_workers=settings['val_workers'])



##############################
# Set Up Things for training #
##############################

if settings['method'] == 'AGL':
    logs_folder = 'logs-adaptive-csv/lightning_logs/'
else:
    logs_folder = 'logs-csv/lightning_logs/'

ver = 0
if 'start_ver' in settings:
    ver = settings['start_ver']
while True:
    # We assume the best model is stored as best.ckpt
    checkpoint = {}
    try:
        for key, val in torch.load(
                logs_folder+f'version_{ver}/checkpoints/best.ckpt',
                map_location=device)['state_dict'].items():
            checkpoint[key.replace('model.', '', 1)] = val
    except FileNotFoundError:
        break

    pl_model = LightningHDNNP_GL(
        dims=settings['dims'], lr=settings['lr'],
        lambda_=0., gamma=settings['gamma'],
        loss=settings['loss'],
        verbose=True, optimizer=settings['optimizer'])

    pl_model.model.load_state_dict(checkpoint)

    masks = pl_model.model.input_mask()

    params = {name: par for name, par in pl_model.model.named_parameters()}
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
    # Now the model should be properly pruned!

    logger_csv = pl.loggers.csv_logs.CSVLogger('logs-continue-csv')
    logger = logger_csv
    checkpointer = pl.callbacks.ModelCheckpoint(monitor='val_rmse',
                                                save_last=True,
                                                save_top_k=5,
                                                filename='{epoch}-{val_rmse:.8f}')
    callbacks = [checkpointer]
    early_stopper = pl.callbacks.EarlyStopping(
        monitor='val_rmse', min_delta=settings['min_delta'], patience=settings['patience'])
    callbacks.append(early_stopper)

    trainer = pl.Trainer(accelerator=accelerator, logger=logger,
                         max_epochs=settings['epochs_adaptive'],
                         enable_progress_bar=False,
                         callbacks=callbacks,
                         track_grad_norm=2,
                         max_time=settings['max_time'])
    trainer.fit(pl_model, train_dl, val_dl)

    ver += 1
