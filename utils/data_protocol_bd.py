# Standard Library
import os
from glob import glob
import pandas as pd

# External Libraries
import buteo as beo
import numpy as np
import json
import config_geography

REGIONS_BUILDINGS = ['DNK', 'EGY', 'GHA', 'ISR', 'TZA', 'UGA']
REGIONS_DOWNSTREAM_DATA = ['denmark-1', 'denmark-2', 'east-africa', 'egypt-1', 'eq-guinea', 'europe', 'ghana-1', 'isreal-1', 'isreal-2', 'japan', 'nigeria','north-america', 'senegal', 'south-america', 'tanzania-1','tanzania-2', 'tanzania-3', 'tanzania-4', 'tanzania-5', 'uganda-1']

REGIONS = REGIONS_DOWNSTREAM_DATA + REGIONS_BUILDINGS


def sanity_check_labels_exist(x_files,y_files, y):
        existing_x = [] 
        existing_y = []
        counter_missing = 0

        assert len(x_files)==len(y_files)
        for x_path,y_path in zip(x_files,y_files):
            
            exists = os.path.exists(y_path)
            if exists:
                existing_x.append(x_path)
                existing_y.append(y_path)
            else:
                counter_missing+=1

        if counter_missing>0:
            print(f'WARNING: {counter_missing} label(s) not found')
            missing = [y_f for y_f in y_files if y_f not in existing_y]
            print(f'Showing up to 5 missing files: {missing[:5]}')

        return existing_x, existing_y

def protocol_all(folder: str, y: str= 'y'):
    """
    Loads all the data from the data folder.
    """

    x_train_files = sorted(glob(os.path.join(folder, f"*train_s2.npy")))
    y_train_files = sorted(glob(os.path.join(folder, f"*train_label_{y}.npy")))

    x_val_files = sorted(glob(os.path.join(folder, f"*val_s2.npy")))
    y_val_files = sorted(glob(os.path.join(folder, f"*val_label_{y}.npy")))

    x_test_files = sorted(glob(os.path.join(folder, f"*test_s2.npy")))
    y_test_files = sorted(glob(os.path.join(folder, f"*test_label_{y}.npy")))

    x_train_files,y_train_files =  sanity_check_labels_exist(x_train_files,y_train_files, y)
    x_val_files,y_val_files =  sanity_check_labels_exist(x_val_files,y_val_files, y)
    x_test_files,y_test_files =  sanity_check_labels_exist(x_test_files,y_test_files, y)

    x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
    y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files])

    x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
    y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files])

    x_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_test_files])
    y_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_test_files])

    assert len(x_train) == len(y_train) and len(x_test) == len(y_test) and len(x_val) == len(
        y_val), "Lengths of x and y do not match."

    return x_train, y_train, x_val, y_val, x_test, y_test


def protocol_split(folder: str, split_percentage: float = 0.2, regions: list = None, y: str = 'y'):
    '''
    Loads a percentage of the data from specified geographic regions.
    '''

    if regions is None:
        regions = REGIONS
    else:
        for r in regions:
            assert r in REGIONS, f"region {r} not found"

    assert 0 < split_percentage <= 1, "split percentage out of range (0 - 1)"

    df = pd.read_csv(glob(os.path.join(folder, f"*.csv"))[0])
    df = df.sort_values(by=['samples'])

    x_train_files = []
    x_test_files = []
    y_test_files = []

    for region in regions:
        mask = [region in f for f in df.iloc[:, 0]]
        df_temp = df[mask].copy().reset_index(drop=True)
        # skip iteration if Region does not belong to current dataset
        if df_temp.shape[0] == 0:
            continue

        df_temp['cumsum'] = df_temp['samples'].cumsum()

        # find row with closest value to the required number of samples
        idx_closest = df_temp.iloc[
            (df_temp['cumsum'] - int(df_temp['samples'].sum() * split_percentage)).abs().argsort()[:1]].index.values[0]
        x_train_files = x_train_files + list(df_temp.iloc[:idx_closest, 0])

        # get test samples of region
        x_test_files = x_test_files + sorted(glob(os.path.join(folder, f"{region}*test_s2.npy")))
        y_test_files = y_test_files + sorted(glob(os.path.join(folder, f"{region}*test_label_{y}.npy")))

    x_train_files = [os.path.join(folder, f_name) for f_name in x_train_files]
    y_train_files = [f_name.replace('s2', f'label_{y}') for f_name in x_train_files]
    x_val_files = [f_name.replace('train', 'val') for f_name in x_train_files]
    y_val_files = [f_name.replace('train', 'val') for f_name in y_train_files]
    
    x_train_files,y_train_files =  sanity_check_labels_exist(x_train_files,y_train_files, y)
    x_val_files,y_val_files =  sanity_check_labels_exist(x_val_files,y_val_files, y)
    x_test_files,y_test_files =  sanity_check_labels_exist(x_test_files,y_test_files, y)

    x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
    y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files])

    x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
    y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files])

    x_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_test_files])
    y_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_test_files])

    assert len(x_train) == len(y_train) and len(x_test) == len(y_test) and len(x_val) == len(
        y_val), "Lengths of x and y do not match."

    return x_train, y_train, x_val, y_val, x_test, y_test


def protocol_regions(folder: str, regions: list = None, y: str = 'y'):
    """
    Loads all the data from the data folder from specified geographic regions.
    """

    if regions is None:
        regions = REGIONS
    else:
        for r in regions:
            assert r in REGIONS, f"region {r} not found"


    x_train_files = []
    y_train_files = []
    x_val_files = []
    y_val_files = []
    x_test_files = []
    y_test_files = []

    for region in regions:
        x_train_files = x_train_files + sorted(glob(os.path.join(folder, f"{region}*train_s2.npy")))
        y_train_files = y_train_files + sorted(glob(os.path.join(folder, f"{region}*train_label_{y}.npy")))

        x_val_files = x_val_files + sorted(glob(os.path.join(folder, f"{region}*val_s2.npy")))
        y_val_files = y_val_files + sorted(glob(os.path.join(folder, f"{region}*val_label_{y}.npy")))

        x_test_files = x_test_files + sorted(glob(os.path.join(folder, f"{region}*test_s2.npy")))
        y_test_files = y_test_files + sorted(glob(os.path.join(folder, f"{region}*test_label_{y}.npy")))


    x_train_files,y_train_files =  sanity_check_labels_exist(x_train_files,y_train_files, y)
    x_val_files,y_val_files =  sanity_check_labels_exist(x_val_files,y_val_files, y)
    x_test_files,y_test_files =  sanity_check_labels_exist(x_test_files,y_test_files, y)

    x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
    y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files])

    x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
    y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files])

    x_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_test_files])
    y_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_test_files])

    assert len(x_train) == len(y_train) and len(x_test) == len(y_test) and len(x_val) == len(
        y_val), "Lengths of x and y do not match."

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
    protocol_split('/home/lcamilleri/data/s12_buildings/data_patches/')
