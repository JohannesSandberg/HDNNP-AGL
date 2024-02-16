#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:03:21 2022

@author: sandberj
"""
# Implementation of HDNNP model in pytorch.

from typing import Callable

import torch
from torch import nn
import itertools as it


class NNP(nn.Module):
    """
    Atomic NNP.

    Representation of an NNP for a single atomic species.

    """

    def __init__(self, dims: tuple, element: str,
                 act: Callable = torch.tanh):

        super().__init__()

        self.element = element
        self.dims = dims
        self.act = act

        self.layers = nn.ModuleList([
            nn.Linear(w_in, w_out)
            for w_in, w_out in zip(dims[:-1], dims[1:])
            ])

    def forward(self, x):
        """
        Apply NNP.

        Applies the NNP along the last axis of the input (feaure axis).
        Note that no axes are summed over, apart from the feature axis.
        If x contains multiple samples along an axis enumerating a batch,
        or an axis enumerating atoms along a configuration, these will not be
        summed over.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the NNP.

        Returns
        -------
        torch.Tensor
            The outputs of the NNP. Atomic contributions to the total energy.

        """
        # Create input mask to remove non-existent atoms
        # TODO! Do a second pass on this. Feels like it might be shortened
        with torch.no_grad():
            mask = torch.logical_not(torch.isnan(x))
            mask = torch.all(mask, dim=-1)
            mask = torch.unsqueeze(mask, dim=-1)

        # Apply network after removing nans
        x = torch.nan_to_num(x)
        for h in self.layers[:-1]:
            x = self.act(h(x))
        x = self.layers[-1](x)

        # Remove non-existing atoms
        return x*mask

    def l2_norm_input(self):
        """
        L2 norm of input weights.

        Returns
        -------
        torch.Tensor.
            The L2 norm of the NNP input weights.

        """
        return torch.norm(self.layers[0].weight, p=2)**2

    def l2_norm_internal(self):
        """
        L2 norm, excluding input weights.

        Returns
        -------
        torch.Tensor.
            The L2 norm of the NNP internal weights.

        """
        return sum(torch.norm(layer.weight, p=2)**2
                   for layer in self.layers[1:])

    def l2_norm(self):
        """
        L2 norm of the NNP.

        Returns
        -------
        torch.Tensor.
            The L2 norm of the NNP weights.

        """
        return sum(torch.norm(layer.weight, p=2)**2
                   for layer in self.layers)

    def group_norm(self):
        """
        Euclidean norm of each input weight group.

        Note that this returns the penalty per input, i.e. without summing
        over inputs. This is to facilitate easier implementation of adaptive
        schemes.
        To get the full GL penalty, sum this over inputs and atoms.

        Returns
        -------
        torch.Tensor.
            The euclidean norm of each input weight group, defined as the
            weights connecting the same input to the first hidden layer.

        """
        return torch.norm(self.layers[0].weight, dim=0).detach()

    def input_mask(self):
        """
        Create a mask over the active inputs.

        Returns
        -------
        torch.Tensor.
            A boolean tensor of  which input features are active, as defined
            by their group-penalty being nonzero.
            TODO: Make sure that the shape allows for convenient broadcasting.

        """
        with torch.no_grad():
            return self.group_norm() != 0

    ######################
    # Proximal Operators #
    ######################
    def apply_proximal(self, lr, lambda_, norm=1):
        """
        Apply the Group Lasso proximal operator.

        Parameters
        ----------
        lr : float
            Current learning rate.
        lambda_ : float
            Penalty strength.
        norm : torch.Tensor, optional
            1D tensor containing multiplicative rescaling factors for lambda_.
            For Adaptive Group Lasso element i should be the inverse of the
            group norm for the i'th input feature.
            If not specified, all input features will have same penalty
            strength, corresponding to non-adaptive Group Lasso.
            A single scalar value can also be passed, in which case it will
            serve as a constant rescaling across all the features.

        """
        with torch.no_grad():
            w = self.layers[0].weight
            tmp = torch.norm(w, dim=0) - lambda_*lr*norm
            tmp = torch.clamp(tmp, min=0)
            tmp = torch.nn.functional.normalize(w, dim=0)*tmp[None, :]
            w.copy_(tmp)

    def detach_state_dict(self):
        """With deepcopied parameters, on cpu, to avoid side effects"""
        return {key: par.detach().clone().cpu()
                for key, par in self.state_dict().items()}

class HDNNP(nn.Module):
    """
    Updated HDNNP class.

    Based on ModuleDicts and new NNP class.
    """

    def __init__(self, dims, act=torch.tanh):
        """
        Create HDNNP.

        Parameters
        ----------
        dims : dict(string: tuple(int))
            A dict mapping element strings (e.g. 'Al', 'B', 'Fe', etc.)
            to a tuple defining the shape of an NNP for that element.
        act : function, optional
            Activation function used in the atomic NNPs.
            By default tanh.

        """
        super().__init__()

        self.dims = dims
        self.elements = list(dims.keys())
        self.act = act

        # Create NNPs
        self.networks = nn.ModuleDict({
            key: NNP(shape, key, act=act) for key, shape in dims.items()
            })

    def forward(self, x):
        """
        Apply HDNNP.

        Parameters
        ----------
        x : dict(string: torch.Tensor)
            A dictionary mapping element strings to tensor inputs for the
            NNP corresponding to that element.

        Returns
        -------
        torch.Tensor.
            The total energy as predicted for by the NNPs, summed over atoms.

        """
        return sum(self.networks[key](val).sum(dim=-2)
                   for key, val in x.items())

    def l2_norm_input(self):
        """
        L2 norm of input weights.

        Calulate the input norm of all the input weights across all
        atomic NNPs.

        Returns
        -------
        torch.Tensor.
            The L2 norm of input weights.

        """
        return sum(network.l2_norm_input()
                   for network in self.networks.values())

    def l2_norm_internal(self):
        """
        L2 norm, excluding input weights.

        Calulate the input norm of all the internal weights across all
        atomic NNPs.

        Returns
        -------
        torch.Tensor.
            The L2 norm of internal weights.

        """
        return sum(network.l2_norm_internal()
                   for network in self.networks.values())

    def l2_norm(self):
        """
        L2 norm of all HDNNP weights.

        Calulate the input norm of all weights across all
        atomic NNPs.

        Returns
        -------
        torch.Tensor.
            The L2 norm of the entire HDNNP.

        """
        return sum(network.l2_norm()
                   for network in self.networks.values())

    def group_norm(self):
        """
        Euclidean norm of all input weights.

        Calculate the euclidean norm of all input weight groups.

        Returns
        -------
        dict(string: torch.Tensor).
            A dictionary mapping element strings to the euclidean norms of
            the input groups of their corresponding NNP.

        """
        return {key: network.group_norm()
                for key, network in self.networks.items()}

    def inverse_group_norm(self):
        """
        Inverse Euclidean norm of all input weights.

        Calculate the (euclidean norm)^-1 of all input weight groups.

        Returns
        -------
        dict(string: torch.Tensor).
            A dictionary mapping element strings to the inverse euclidean norms
            of the input groups of their corresponding NNP.

        """
        return {key: torch.nan_to_num(1/network.group_norm())
                for key, network in self.networks.items()}

    def input_mask(self):
        """
        Create input masks over active inputs.

        Returns
        -------
        dict.
            A dictionary mapping element strings to the input masks of their
            corresponding NNP.

        """
        return {key: network.input_mask()
                for key, network in self.networks.items()}

    def apply_proximal(self, lr, lambda_, norm=1):
        """
        Apply the Group Lasso proximal operator.

        Parameters
        ----------
        lr : float
            Current learning rate.
        lambda_ : float
            Penalty strength.
        norm : dict, optional
            Dictionary mapping element strings to tensors of 1D multiplicative
            rescaling factors for lambda_.
            Calls NNP.apply_proximal on each NNP with the factors mapped
            to by the same element string. For more information, see the
            documentation for that method.
            By default, no rescaling is performed.
            If a single scalar value is passed instead of a dict, then it
            will serve as a constant rescaling across all features, across
            all NNPs.

        """
        if isinstance(norm, dict):
            for key, network in self.networks.items():
                network.apply_proximal(lr, lambda_, norm[key])
            return
        # If norm is not a dict, we assume it is a constant scale
        for network in self.networks.values():
            network.apply_proximal(lr, lambda_, norm)

    def detach_state_dict(self):
        """With deepcopied parameters, on cpu, to avoid side effects"""
        return {key: par.detach().clone().cpu()
                for key, par in self.state_dict().items()}


def save_N2P2(model, folder):
    """Save High-Dimensional model in N2P2 format."""
    if not folder.endswith('/'):
        folder += '/'
    header = '################################################################################\n'
    header += '# Neural network connection values (weights and biases).\n'
    header += '################################################################################\n'
    header += '# Col  Name       Description\n'
    header += '################################################################################\n'
    header += '# 1    connection Neural network connection value.\n'
    header += '# 2    t          Connection type (a = weight, b = bias).\n'
    header += '# 3    index      Index enumerating weights.\n'
    header += '# 4    l_s        Starting point layer (end point layer for biases).\n'
    header += '# 5    n_s        Starting point neuron in starting layer (end point neuron for biases).\n'
    header += '# 6    l_e        End point layer.\n'
    header += '# 7    n_e        End point neuron in end layer.\n'
    header += '################################################################################\n'
    header += '#                      1 2         3     4     5     6     7\n'
    header += '#             connection t     index   l_s   n_s   l_e   n_e\n'
    header += '############################################################\n'
    for element, network in model.networks.items():
        with open(folder+'weights.'+element+'.data', 'x') as f:
            f.write(header)
            idx = 1
            for l,layer in enumerate(network.layers):
                NOut, NIn = layer.weight.shape
                w = layer.weight.detach().cpu()
                for n_s, n_e in it.product(range(1, NIn+1), range(1, NOut+1)):
                    f.write(' {: 23.16E} a {: >9} {: >5} {: >5} {: >5} {: >5}\n'.format(
                        w[n_e-1, n_s-1], idx, l, n_s, l+1, n_e))
                    idx += 1
                b = layer.bias.detach().cpu()
                for n_e in range(1, NOut+1):
                    f.write(' {: 23.16E} b {: >9} {: >5} {: >5}\n'.format(
                        b[n_e-1], idx, l+1, n_e))
                    idx += 1
