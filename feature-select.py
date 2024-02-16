#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################
# Feature Selection with Lightning #
####################################
# Options, in addition to standard training ones
# -- method: 'GL' or 'AGL'
# -- warm_restarts: True, False
# -- lambda: float, list
# -- other options
# -- Optional Starting Model (for warm restarts)
import sys
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt

import yaml

import torch
import pytorch_lightning as pl

from torchmetrics import MeanSquaredError

from hdnnpy.model import HDNNP
from hdnnpy.data import prepare_dataset
from hdnnpy.lightning_models import LightningHDNNP_GL

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

########################
# Load or Create model #
########################
if 'initial_model' in settings:
    # Load old model weights

    checkpoint = {}
    for key, val in torch.load(
            settings['initial_model'],
            map_location=device)['state_dict'].items():
        checkpoint[key.replace('model.', '', 1)] = val

if isinstance(settings['lambda'], float) or isinstance(settings['lambda'], int):
    settings['lambda'] = [settings['lambda']]

for lambda_ in settings['lambda']:
    lambda_ = float(lambda_)
    if settings['method'] == 'GL' or settings['method'] == 'AGL':
        pl_model = LightningHDNNP_GL(
            dims=settings['dims'], lr=settings['lr'],
            lambda_=lambda_, gamma=settings['gamma'],
            loss=settings['loss'],
            verbose=True, optimizer=settings['optimizer'])
    if settings['warm_restarts']:
        pl_model.model.load_state_dict(checkpoint)
    # Train
    logger_csv = pl.loggers.csv_logs.CSVLogger('logs-csv')
    logger = logger_csv

    checkpointer = pl.callbacks.ModelCheckpoint(monitor='val_objective',
                                                save_last=True,
                                                save_top_k=5,
                                                filename='{epoch}-{val_objective:.8f}')
    callbacks = [checkpointer]
    early_stopper = pl.callbacks.EarlyStopping(
        monitor='val_objective', min_delta=settings['min_delta'], patience=settings['patience'])
    callbacks.append(early_stopper)
    trainer = pl.Trainer(accelerator=accelerator, logger=logger,
                         max_epochs=settings['epochs'],
                         enable_progress_bar=False,
                         callbacks=callbacks,
                         track_grad_norm=2,
                         max_time=settings['max_time'])
    trainer.fit(pl_model, train_dl, val_dl)

    pl_model.model.load_state_dict(pl_model.best_dict)
    # If adaptive we should get the the norm of the best performing model,
    # reload the initial model, and retrain
    if settings['method'] == 'AGL':
        norm = pl_model.model.inverse_group_norm()
        pl_model = LightningHDNNP_GL(
            dims=settings['dims'], lr=settings['lr'],
            lambda_=lambda_, gamma=settings['gamma'],
            loss=settings['loss'],
            verbose=True,
            norm=norm, optimizer=settings['optimizer'])
        if settings['warm_restarts']:
            pl_model.model.load_state_dict(checkpoint)
        logger_csv = pl.loggers.csv_logs.CSVLogger('logs-adaptive-csv')
        logger = logger_csv

        checkpointer = pl.callbacks.ModelCheckpoint(monitor='val_objective',
                                                    save_last=True,
                                                    save_top_k=5,
                                                    filename='{epoch}-{val_objective:.8f}')
        callbacks = [checkpointer]
        early_stopper = pl.callbacks.EarlyStopping(
            monitor='val_objective', min_delta=settings['min_delta'], patience=settings['patience'])
        callbacks.append(early_stopper)
        trainer = pl.Trainer(accelerator=accelerator, logger=logger,
                             max_epochs=settings['epochs_adaptive'],
                             enable_progress_bar=False,
                             callbacks=callbacks,
                             track_grad_norm=2,
                             max_time=settings['max_time'])
        trainer.fit(pl_model, train_dl, val_dl)

        pl_model.model.load_state_dict(pl_model.best_dict)

    if settings['warm_restarts']:
        checkpoint = pl_model.model.detach_state_dict()

    if pl_model.get_n_features() == 0:
        break

