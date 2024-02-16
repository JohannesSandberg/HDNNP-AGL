#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:12:36 2023

A collection of Pytorch Lightning LightningModules for use in training
HDNNPs, ready to be imported or copy-pasted as needed.

@author: sandberj
"""
import torch
import pytorch_lightning as pl

from torchmetrics import MeanSquaredError

from .model import HDNNP

############################################################
# With Warm Restarts Scheduler, for use with AdamW or SGD, #
# and MSE of Huber loss functions.                         #
############################################################
class LightningHDNNP(pl.LightningModule):
    """
    Pytorch Lightning version of HDNNP.

    Uses Cosine Annealing Warm Restarts LR scheduler.
    Can use MSE Loss or Huber Loss, with either AdamW or SGD optimizer.

    Logs the loss function, the MSE, and RMSE metrics.
    Predicts total energy of system, but loss and metrics are calculated
    with total energy divided by number of atoms.
    """
    def __init__(self, dims, lr, loss='mse', optimizer='SGD', gamma=0.,
                 T_0=80, T_mult=1, min_lr=0., act=torch.tanh):
        super().__init__()
        self.model = HDNNP(dims, act=act)
        if loss.casefold() == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        elif loss.casefold() == 'huber':
            self.loss_fn = torch.nn.HuberLoss()

        # Metrics
        self.train_mse = MeanSquaredError()
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_mse = MeanSquaredError()
        self.val_rmse = MeanSquaredError(squared=False)

        self.save_hyperparameters()

    def forward(self, X):
        return self.model(X)

    def prepare_batch(self, batch):
        X, y, N = batch[0], batch[1], batch[2]
        X = {key: x.to(self.device, non_blocking=True) for key, x in X.items()}
        y = y.to(self.device, non_blocking=True)
        N = N.to(self.device, non_blocking=True)

        return X, y, N

    def training_step(self, batch, batch_idx):
        X, y, N = self.prepare_batch(batch)

        y_pred = self(X)/N
        y_norm = y/N
        loss = self.loss_fn(y_pred, y_norm)
        self.train_mse(y_pred, y_norm)
        self.train_rmse(y_pred, y_norm)

        self.log('train_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log('train_mse', self.train_mse, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log('train_rmse', self.train_rmse, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y, N = self.prepare_batch(batch)

        y_pred = self(X)/N
        y_norm = y/N
        loss = self.loss_fn(y_pred, y_norm)
        self.val_mse(y_pred, y_norm)
        self.val_rmse(y_pred, y_norm)
        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log('val_mse', self.val_mse, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        # RMSE should be computed from MSE at end of each epoch!
        self.log('val_rmse', self.val_rmse, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

    def configure_optimizers(self):
        if self.hparams.optimizer.casefold() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.hparams.lr,
                                          weight_decay=self.hparams.gamma)
        elif self.hparams.optimizer.casefold() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.hparams.lr,
                                        weight_decay=self.hparams.gamma)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.hparams.T_0,
            T_mult=self.hparams.T_mult,
            eta_min=self.hparams.min_lr)

        scheduler_config = {'scheduler': scheduler,
                            'interval': 'step',
                            'frequency': 1}

        return {'optimizer': optimizer,
                'lr_scheduler': scheduler_config}


class LightningHDNNP_GL(pl.LightningModule):
    """
    Pytorch Lightning version of HDNNP with GroupLasso.

    Can use MSE Loss or Huber Loss, SGD optimizer.

    Logs the loss function, the MSE, and RMSE metrics.
    Predicts total energy of system, but loss and metrics are calculated
    with total energy divided by number of atoms.

    Values of norm should be tensors of shape such that they broadcasts
    correctly with the features of corresponding .
    Ideally related to pl_model.model.group_norm().
    """

    def __init__(self, dims, lr, lambda_, loss='mse', gamma=0., gamma_input=0.,
                 norm=None, verbose=False, optimizer='SGD', act=torch.tanh):
        super().__init__()
        self.model = HDNNP(dims, act=act)
        if loss.casefold() == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        elif loss.casefold() == 'huber':
            self.loss_fn = torch.nn.HuberLoss()

        # Metrics
        self.train_mse = MeanSquaredError()
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_mse = MeanSquaredError()
        self.val_rmse = MeanSquaredError(squared=False)
        self.save_hyperparameters()
        if norm is None:
            self.adaptive=False
            self.norm = {el: torch.tensor(1) for el in dims}
        else:
            self.adaptive=True
            self.norm = norm
        self.best_obj = torch.inf
        self.best_dict = self.model.detach_state_dict()
        self.verbose=verbose
        self.n_features = sum(d[0] for d in dims.values())

    def on_fit_start(self):
        # In __init__ the model is still on cpu.
        # Here it will have been moved to device (e.g. gpu), and so we
        # move the norm to device here so that we only need to move it once.
        self.norm = {el:self.norm[el].to(self.device) for el in self.norm}

    def forward(self, X):
        return self.model(X)

    def prepare_batch(self, batch):
        X, y, N = batch[0], batch[1], batch[2]
        X = {key: x.to(self.device, non_blocking=True) for key, x in X.items()}
        y = y.to(self.device, non_blocking=True)
        N = N.to(self.device, non_blocking=True)

        return X, y, N

    def on_train_batch_start(self,batch,batch_idx):
        if self.n_features == 0:
            return -1

    def training_step(self, batch, batch_idx):
        X, y, N = self.prepare_batch(batch)

        y_pred = self(X)/N
        y_norm = y/N
        loss = self.loss_fn(y_pred, y_norm)
        self.train_mse(y_pred, y_norm)
        self.train_rmse(y_pred, y_norm)

        self.log('train_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log('train_mse', self.train_mse, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log('train_rmse', self.train_rmse, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

        return loss

    def on_train_batch_end(self, out, batch, batch_idx):
        with torch.no_grad():
            self.model.apply_proximal(self.hparams.lr,
                                      self.hparams.lambda_,
                                      self.norm)

    def get_n_features(self):
        return sum(w.sum().data
                         for w in self.model.input_mask().values())

    def on_validation_epoch_end(self):
        with torch.no_grad():
            regularization = self.hparams.lambda_*sum(
                self.norm[el]*reg
                for el, reg in self.model.group_norm().items()).sum()
            objective = self.val_mse.compute() + regularization
            self.n_features = self.get_n_features()
            self.log('n_features', self.n_features, on_step=False,
                     on_epoch=True, prog_bar=True, logger=True)
            self.log('val_objective', objective, on_step=False,
                     on_epoch=True, prog_bar=True, logger=True)
            if objective < self.best_obj:
                rmse = self.val_rmse.compute()
                print(f'adaptive:{self.adaptive}, lambda:{self.hparams.lambda_:.2e}, Epoch: {self.current_epoch}, VAL RMSE:{rmse:.8f}, VAL OBJECTIVE:{objective:.8f}, VAL MSE:{rmse**2:.8f}, NFeatures:{self.n_features}')
                self.best_obj = objective
                self.best_dict = self.model.detach_state_dict()

            group_norm = self.model.group_norm()
            for el, norm in group_norm.items():
                self.log_dict(
                    {f'{el}-input-{i}': x for i, x in enumerate(norm)},
                    on_epoch=True, on_step=False)


    def validation_step(self, batch, batch_idx):
        X, y, N = self.prepare_batch(batch)

        y_pred = self(X)/N
        y_norm = y/N
        loss = self.loss_fn(y_pred, y_norm)
        self.val_mse(y_pred, y_norm)
        self.val_rmse(y_pred, y_norm)
        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log('val_mse', self.val_mse, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        # RMSE should be computed from MSE at end of each epoch!
        self.log('val_rmse', self.val_rmse, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)


    def configure_optimizers(self):
        weights_input = []
        weights_internal = []
        biases = []
        for el in self.model.networks:
            for idx, layer in enumerate(self.model.networks[el].layers):
                biases.append({'params': layer.bias, 'weight_decay':0.})
                if idx==0:
                    weights_input.append(
                        {'params': layer.weight,
                         'weight_decay': self.hparams.gamma_input})
                else:
                    weights_internal.append({'params': layer.weight,
                                             'weight_decay': self.hparams.gamma})
        params = weights_input+weights_internal+biases

        if self.hparams.optimizer.casefold() == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)
        elif self.hparams.optimizer.casefold() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.hparams.lr)

        return {'optimizer': optimizer}
