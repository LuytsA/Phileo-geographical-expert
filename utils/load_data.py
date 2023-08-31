# Standard Library
import os
from glob import glob

# External Libraries
import buteo as beo
import numpy as np

# PyTorch
import torch
from torch.utils.data import DataLoader
from utils.training_utils import encode_coordinates


def callback_preprocess(x, y):
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    y = y.astype(np.float32, copy=False)

    return x_norm, y

def callback_postprocess_encoder(x, y):
    x = beo.channel_last_to_first(x)
    #y = np.array([np.sum(y) / y.size], dtype=np.float32)
    y_coords = y[:2]
    y_coords = encode_coordinates(y_coords)
    y = np.concatenate([y_coords, y[2:]],dtype=np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)

def callback_postprocess_decoder(x, y):
    x = beo.channel_last_to_first(x)
    y = beo.channel_last_to_first(y)

    return torch.from_numpy(x), torch.from_numpy(y)

def callback_encoder(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess_encoder(x, y)

    return x, y

def callback_decoder(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def load_data(x_train, y_train, x_val, y_val, x_test, y_test, with_augmentations=False, num_workers=0, batch_size=16, encoder_only=False):
    """
    Loads the data from the data folder.
    """
    
    if with_augmentations:
        ds_train = beo.DatasetAugmentation(
            x_train, y_train,
            callback_pre_augmentation=callback_preprocess,
            callback_post_augmentation=callback_postprocess_encoder if encoder_only else callback_postprocess_decoder,
            augmentations=[
                beo.AugmentationRotationXY(p=0.2, inplace=True),
                beo.AugmentationMirrorXY(p=0.2, inplace=True),
                beo.AugmentationCutmix(p=0.2, inplace=True),
                beo.AugmentationNoiseNormal(p=0.2, inplace=True),
            ]
        )
    else:
        ds_train = beo.Dataset(x_train, y_train, callback=callback_encoder if encoder_only else callback_decoder)

    ds_test = beo.Dataset(x_test, y_test, callback=callback_encoder if encoder_only else callback_decoder)
    ds_val = beo.Dataset(x_val, y_val, callback=callback_encoder if encoder_only else callback_decoder)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True, generator=torch.Generator(device='cuda'))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True, generator=torch.Generator(device='cuda'))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True, generator=torch.Generator(device='cuda'))

    return dl_train, dl_val, dl_test
