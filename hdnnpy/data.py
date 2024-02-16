#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:08:19 2022

@author: sandberj
"""
# Various methods for loading and manipulating data for HDNNP, in pytorch
# Should to as large an extent as possible use Lassonet Dataset functionality,
# and also allow us to preprocess the data as appropriate

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from time import time
from sklearn.model_selection import train_test_split, KFold
import os
import sys
# import itertools as it
from copy import deepcopy

def collate_fn(data):
    X, y, N = zip(*data)
    X = {key: torch.nn.utils.rnn.pad_sequence(
        [d[key] for d in X], batch_first=True, padding_value=np.nan)
        for key in X[0]}

    return X, torch.stack(y).reshape(-1, 1), torch.stack(N).reshape(-1, 1)


class Config:
    """
    Container for a single atomic configuration.
    """

    def __init__(self, energy, fingerprints):
        """
        Single configuration.

        Parameters
        ----------
        energy : torch tensor
            Energy of the configuration.
        fingerprints : dict
            Dictionary mapping element strings to a torch tensor containing
            the atomic fingerprints of atoms of that type, for the given
            atomic configuration.

        Returns
        -------
        None.

        """
        self.energy = energy
        self.fingerprints = fingerprints
        self.NAtoms = torch.tensor(sum(len(val)
                                       for val in fingerprints.values()))

    def get_data(self):
        """Get fingerprints and energy of configuration."""
        return self.fingerprints, self.energy, self.NAtoms

    def set_fingerprints(self, fingerprints):
        self.fingerprints = fingerprints

class ScalingData:
    def __init__(self, mean, stdev, min_, max_):
        """
        Construct Scaling Data.

        Note: All tensors should be of shape (1,NFingerprints),
            i.e. with atom axis of length 1, to ensure proper
            broadcasting.

        Parameters
        ----------
        mean : dict
            Dictionary mapping element strings to a tensor of
            mean input values.
        stdev : dict
            Dictionary mapping element strings to a tensor of
            standard deviations of input values.
        min_ : dict
            Dictionary mapping element strings to a tensor of
            minimum input values.
        max_ : dict
            Dictionary mapping element strings to a tensor of
            maximum input values.

        """
        self.mean = mean
        self.stdev = stdev
        self.min = min_
        self.max = max_
        self.elements = list(mean)

    def scale_standard(self, G):
        """Apply standard scaling."""
        return {key: (g - self.mean[key])/self.stdev[key]
                for key, g in G.items()}

    def scale_minmax(self, G):
        """Apply min-max scaling."""
        return {key: (2.*g - (self.max[key] - self.min[key]))
                / (self.max[key] - self.min[key])
                for key, g in G.items()}

    def scale_scale(self, G):
        """Scale by standard deviation."""
        return {key: g/self.stdev[key]
                for key, g in G.items()}

    def scale_shift(self, G):
        """Scale by shifting to zero mean."""
        return {key: g - self.mean[key]
                for key, g in G.items()}

    def unscale_standard(self, G):
        """Undo scaling."""
        return {key: (g * self.stdev[key]) + self.mean[key]
                for key, g in G.items()}

    def unscale_minmax(self, G):
        """Undo scaling."""
        return {key: ((g * (self.max[key] - self.min[key]))
                      + (self.max[key] - self.min[key]))/2.
                for key, g in G.items()}

    def unscale_scale(self, G):
        """Undo scaling."""
        return {key: g * self.stdev[key]
                for key, g in G.items()}

    def unscale_shift(self, G):
        """Undo scaling."""
        return {key: g + self.mean[key]
                for key, g in G.items()}


class HDNNP_Dataset(Dataset):

    def __init__(self, atomenv, input_data, labels):
        """
        HDNNP dataset.

        Note that the dataset is fully stored in memory, and as such this has
        a large memory footprint.

        Parameters
        ----------
        atomenv : string
            File from which to read the atomic fingerprints.
        input_data : string
            input.data file used to generate the fingerprints.
        labels : dict
            Dictionary mapping element strings to a list of fingerprint labels
            for that element.
            Should ideally describe the parameters of the fingerprints.
            We suggest that it be taken from nnp-atomenv.log or a similar file,
            ensuring consistency with the internal ordering n2p2.

        """
        self.source = (atomenv, input_data)
        self.labels = labels
        self.NSpecies = len(labels)

        ###########################################
        # Read files and construct configurations #
        ###########################################
        configs = []
        with open(atomenv, 'r') as f_G:
            with open(input_data, 'r') as f_E:
                fingerprints = {key: [] for key in labels}
                for line in f_E:
                    cols = line.lstrip().split()
                    if cols[0] == 'atom':
                        # Each config will have NAtom lines
                        # beginning with atom, followed by
                        # 3 atomic coordinates, followed by
                        # 2 useless items, followed by
                        # 1 atomic element string, followed by
                        # 3 forces
                        G = f_G.readline().lstrip().split()
                        # G[0] is element
                        # G[1:] is the fingerprints
                        fingerprints[G[0]].append(
                            torch.tensor([float(g) for g in G[1:]]))
                    if cols[0] == 'energy':
                        energy = torch.tensor(float(cols[1]))
                    if cols[0] == 'end':
                        # Convert fingerprint arrays to tensors
                        for key, G in fingerprints.items():
                            if len(G) == 0:
                                # No atoms of type key i configuration
                                # Should be shape [0, len(labels[key])]
                                fingerprints[key] = torch.tensor([]).reshape(
                                    [0, len(labels[key])])
                                continue
                            fingerprints[key] = torch.stack(G)
                        configs.append(Config(energy, fingerprints))
                        fingerprints = {key: [] for key in labels}
        self.configs = configs
        self.scaling_type = 'none'
        # When initialized these should be dicts that map element strings
        # to the corresponding tensors, of correct shape (1,NFeatures)
        # to allow for proper broadcasting.
        self.scaling_data = None

    def __getitem__(self, idx):
        return self.configs[idx].get_data()

    def __len__(self):
        return len(self.configs)

    def get_labels(self):
        return self.labels

    def calculate_scaling(self):
        """

        Calculate scaling data.

        Calculate the max, min, mean, and standard deviation of each
        fingerprint across the dataset.

        """
        G_max = {}
        G_min = {}
        means = {}
        stdevs = {}
        for key in self.labels:
            g = torch.vstack(
                [config.fingerprints[key] for config in self.configs])
            G_max[key] = torch.max(g, dim=0, keepdim=True)[0]
            G_min[key] = torch.min(g, dim=0, keepdim=True)[0]
            means[key] = torch.mean(g, dim=0, keepdim=True)
            stdevs[key] = torch.std(g, dim=0, keepdim=True)

        self.scaling_data = ScalingData(means, stdevs, G_min, G_max)


    def set_scaling(self, scaling):
        """

        Set scaling data.

        Set the max, min, mean, and standard deviation of each
        fingerprint via a ScalingData object.

        Parameters
        ----------
        scaling : ScalingData object
        Scaling data to use.

        """
        self.scaling_data = scaling

    def get_scaling(self):
        """
        Get scaling data.
        """
        return self.scaling_data

    def scale(self, scaling_type='standard'):
        """

        Apply scaling to the dataset.

        Applies scaling of the given type to the dataset.
        If scaling is already applied, first unscale.

        Parameters
        ----------
        scaling_type : string
        Valid options include standard, minmax, scale, and shift.
        standard: Shift by mean, and scale by standard deviation.
        minmax: Shift and scale so that range is +-1
        scale: Scale by standard deviation.
        shift: Shift by mean.

        """
        if self.scaling_type == scaling_type:
            return
        if self.scaling_data is None and self.scaling_type is not None:
            sys.exit('ERROR: Scaling data not calculated!')
        if scaling_type not in ['scale', 'minmax',
                                'shift', 'standard', None]:
            sys.exit('ERROR: Invalid scaling type!')
        if not self.scaling_type is None:
            self.unscale()
        for config in self.configs:
            if scaling_type == 'standard':
                config.set_fingerprints(
                    self.scaling_data.scale_standard(config.fingerprints))
            if scaling_type == 'minmax':
                config.set_fingerprints(
                    self.scaling_data.scale_minmax(config.fingerprints))
            if scaling_type == 'scale':
                config.set_fingerprints(
                    self.scaling_data.scale_scale(config.fingerprints))
            if scaling_type == 'shift':
                config.set_fingerprints(
                    self.scaling_data.scale_shift(config.fingerprints))
        self.scaling_type = scaling_type

    def unscale(self):
        """

        Undo scaling.

        Undoes the applied scaling, allowing for accessing the
        unscaled fingerprints, or applying a different scaling type.

        """
        if self.scaling_type == 'none':
            return
        for config in self.configs:
            if self.scaling_type == 'standard':
                config.set_fingerprints(
                    self.scaling_data.unscale_standard(config.fingerprints))
            if self.scaling_type == 'minmax':
                config.set_fingerprints(
                    self.scaling_data.unscale_minmax(config.fingerprints))
            if self.scaling_type == 'scale':
                config.set_fingerprints(
                    self.scaling_data.unscale_scale(config.fingerprints))
            if self.scaling_type == 'shift':
                config.set_fingerprints(
                    self.scaling_data.unscale_shift(config.fingerprints))
        self.scaling_type = 'none'


def split_dataset(dataset, split=0.2, seed=None, shuffle=True):
    """
    Split a HDNNP dataset into two smaller datasets d1 and d2.

    Parameters
    ----------
    dataset : HDNNP_Dataset
        Dataset to be split.
    split : float, optional
        Proportion of the data to be put in d2.
        The default is 0.2.
    seed : int, optional
        Random seed used if shuffling is activated. The default is None.
    shuffle : TYPE, optional
        Whether the dataset is shuffled before splitting.
        The default is True.

    Returns
    -------
    Datasets d1 and d2.

    """
    configs = dataset.configs
    c1, c2 = train_test_split(configs, test_size=split,
                              random_state=seed, shuffle=shuffle)
    d1 = deepcopy(dataset)
    d2 = deepcopy(dataset)
    d1.configs = c1
    d2.configs = c2
    return d1, d2


def read_N2P2_scaling(file, mapping):
    """
    Read N2P2 scaling.dat file.

    Parameters
    ----------
    file : file
        scaling.data file to read from.
    mapping : dict
        Dictionary mapping element indices in the scaling file to
        element strings.

    Returns
    -------
    ScalingData object.

    """
    maxvals = []
    minvals = []
    means = []
    stdevs = []
    species = []
    index = []

    max_dict = {}
    min_dict = {}
    mean_dict = {}
    stdev_dict = {}

    with open(file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.strip().split()
            species.append(int(cols[0]))
            index.append(int(cols[1]))
            minvals.append(float(cols[2]))
            maxvals.append(float(cols[3]))
            means.append(float(cols[4]))
            stdevs.append(float(cols[5]))
    for s in set(species):
        mean_s = torch.tensor(
            [x for i,x in enumerate(means) if species[i] == s])
        index_s = torch.tensor(
            [x for i,x in enumerate(index) if species[i] == s])
        stdev_s = torch.tensor(
            [x for i,x in enumerate(stdevs) if species[i] == s])
        max_s = torch.tensor(
            [x for i,x in enumerate(maxvals) if species[i] == s])
        min_s = torch.tensor(
            [x for i,x in enumerate(minvals) if species[i] == s])

        order = torch.argsort(index_s)
        mean_s = mean_s[order].reshape(1,-1)
        stdev_s = stdev_s[order].reshape(1,-1)
        max_s = max_s[order].reshape(1,-1)
        min_s = min_s[order].reshape(1,-1)

        max_dict[mapping[s]] = max_s
        min_dict[mapping[s]] = min_s
        mean_dict[mapping[s]] = mean_s
        stdev_dict[mapping[s]] = stdev_s
    return ScalingData(mean_dict, stdev_dict, min_dict, max_dict)



def save_N2P2_scaling(file, scaling, mapping=None):
    """
    Save scaling data in the N2P2 scaling.dat format.

    Parameters
    ----------
    file : file
        File to save to.
    scaling : ScalingData
        ScalingData object to save.
    mapping : dictionary
        Mapping from element strings to element indices.
        Optional, by default the elements will be indexed 1,2,3,... by their
        order in the scaling data means.

    """
    header = '################################################################################\n'
    header += '# Symmetry function scaling data.\n'
    header += '################################################################################\n'
    header += '# Col  Name     Description\n'
    header += '################################################################################\n'
    header += '# 1    e_index  Element index.\n'
    header += '# 2    sf_index Symmetry function index.\n'
    header += '# 3    sf_min   Symmetry function minimum.\n'
    header += '# 4    sf_max   Symmetry function maximum.\n'
    header += '# 5    sf_mean  Symmetry function mean.\n'
    header += '# 6    sf_sigma Symmetry function sigma.\n'
    header += '#########################################################################################################################\n'
    header += '#        1          2                        3                        4                        5                        6\n'
    header += '#  e_index   sf_index                   sf_min                   sf_max                  sf_mean                 sf_sigma\n'
    header += '#########################################################################################################################\n'

    with open(file, 'w') as f:
        f.write(header)
        # Iterate over species
        for s,key in enumerate(scaling.elements):
            if mapping is None:
                idx = s
            else:
                idx = mapping[key]
            means = scaling.mean[key]
            stdevs = scaling.stdev[key]
            maxvals = scaling.max[key]
            minvals = scaling.min[key]
            num_features = means.shape[1]
            for i in range(num_features):
                outstring='{: >10}{: >11}{: >25.16E}{: >25.16E}{: >25.16E}{: >25.16E}\n'.format(
                    idx+1,i+1,minvals[0,i],maxvals[0,i],means[0,i],stdevs[0,i])
                f.write(outstring)


def read_labels_N2P2(file):
    """Read labels from an nnp-atomenv log or output file."""
    labels = dict()
    with open(file, 'r') as f:
        read = False
        for line in f:
            cols=line.strip().split()
            if len(cols) <= 1:
                continue
            if line.startswith('Short range atomic symmetry functions element '):
                element = cols[6]
                read = True
                labels[element] = []
            if read and cols[1] == element:
                labels[element].append(line.strip())
            if line.startswith('*'):
                read = False
    return labels


def prepare_dataset(train_data_path, labels_file, bs=256, split=0.2, seed=None,
                    pin_memory=True, verbose=True, train_workers=0,
                    val_workers=0, scaling_file=None):
    """Prepare training and validation dataset for training."""
    # Extract Labels - NOTE: labels_file should be an nnp-atomenv log file
    if verbose:
        print('Extracting SymFun Labels from N2P2 log file:\n    ',
              labels_file)
    labels = read_labels_N2P2(labels_file)
    if verbose:
        print('Symmetry Functions:')
    for key, labels_list in labels.items():
        if verbose:
            print('Element: ', key)
        for symfun in labels_list:
            if verbose:
                print('  ', symfun)
    if verbose:
        print('#'*22)
        print('Loading Dataset Directory: ', train_data_path)

    train_dataset = HDNNP_Dataset(train_data_path+'atomic-env.G',
                                  train_data_path+'input.data',
                                  labels)
    # Calculate scaling data
    if scaling_file is None:
        if verbose:
            print('Calculating Scaling')
        train_dataset.calculate_scaling()
        scaling = train_dataset.get_scaling()
        if verbose:
            print('Saving Scaling to scaling.data')
        save_N2P2_scaling('scaling.data', scaling)
    else:
        scaling = read_N2P2_scaling(scaling_file,
                                    mapping={i+1:el for i,el in enumerate(
                                        train_dataset.labels)})
        train_dataset.set_scaling(scaling)
    train_dataset.scale(scaling_type='standard')

    # Split dataset
    if verbose:
        print('Splitting Dataset')
    train_dataset, val_dataset = split_dataset(train_dataset, split, seed)

    # Prepare dataloaders
    train_dl = DataLoader(train_dataset, batch_size=bs,
                          shuffle=True, collate_fn=collate_fn,
                          pin_memory=pin_memory, num_workers=train_workers)
    val_dl = DataLoader(val_dataset, batch_size=bs,
                        shuffle=False, collate_fn=collate_fn,
                        pin_memory=pin_memory, num_workers=val_workers)

    return train_dl, val_dl
