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
REGIONS_ROADS = ['north-america','east-africa', 'europe','eq-guinea', 'japan','south-america', 'nigeria', 'senegal']
REGIONS = REGIONS_ROADS + REGIONS_BUILDINGS

region_map = config_geography.regions


def protocol_all(folder: str, y: str= 'y'):
    """
    Loads all the data from the data folder.
    """

    if y == 'geography':
        x_train_files = sorted(glob(os.path.join(folder, f"*train_s2.npy")))
        x_val_files = sorted(glob(os.path.join(folder, f"*val_s2.npy")))
        x_test_files = sorted(glob(os.path.join(folder, f"*test_s2.npy")))

        x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
        x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
        x_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_test_files])

        y_part = []
        with open('/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/tiles_coordinates.json') as fp:
            tiles_coord = json.load(fp)
        for part in ['train','test','val']:
            ff = sorted(glob(os.path.join(folder, f"*{part}_s2.npy")))
            tile_names = [f.split('/')[-1].split(f'_{part}_')[0] for f in ff]
            tile_shapes = [np.load(f, mmap_mode='r').shape[0] for f in ff]
            regions = [region_map[f.split('_')[0]] for f in tile_names]
            assert len(tile_names)==len(tile_shapes)
            y_part.append(beo.MultiArray([ np.repeat(np.array([tiles_coord[n]+[r]]),s, axis=0) for n,s,r in zip(tile_names,tile_shapes,regions)]))
        
        y_train, y_test, y_val = y_part

    else:
        x_train_files = sorted(glob(os.path.join(folder, f"*train_s2.npy")))
        y_train_files = sorted(glob(os.path.join(folder, f"*train_label_{y}.npy")))

        x_val_files = sorted(glob(os.path.join(folder, f"*val_s2.npy")))
        y_val_files = sorted(glob(os.path.join(folder, f"*val_label_{y}.npy")))

        x_test_files = sorted(glob(os.path.join(folder, f"*test_s2.npy")))
        y_test_files = sorted(glob(os.path.join(folder, f"*test_label_{y}.npy")))

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



    if y == 'geography':

        x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
        x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
        x_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_test_files])

        y_part = []
        with open('/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/tiles_coordinates.json') as fp:
            tiles_coord = json.load(fp)
        for part, ff in zip(['train','test','val'],[x_train_files,x_test_files,x_val_files]):
            tile_names = [f.split('/')[-1].split(f'_{part}_')[0] for f in ff]
            tile_shapes = [np.load(f, mmap_mode='r').shape[0] for f in ff]
            regions = [region_map[f.split('_')[0]] for f in tile_names]
            assert len(tile_names)==len(tile_shapes)
            y_part.append(beo.MultiArray([ np.repeat(np.array([tiles_coord[n]+[r]]),s, axis=0) for n,s,r in zip(tile_names,tile_shapes,regions)]))
        
        y_train, y_test, y_val = y_part

    else:
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


    if y == 'geography':

        x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
        x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
        x_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_test_files])

        y_part = []
        with open('/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/tiles_coordinates.json') as fp:
            tiles_coord = json.load(fp)
        for part, ff in zip(['train','test','val'],[x_train_files,x_test_files,x_val_files]):
            tile_names = [f.split('/')[-1].split(f'_{part}_')[0] for f in ff]
            tile_shapes = [np.load(f, mmap_mode='r').shape[0] for f in ff]
            regions = [region_map[f.split('_')[0]] for f in tile_names]
            assert len(tile_names)==len(tile_shapes)
            y_part.append(beo.MultiArray([ np.repeat(np.array([tiles_coord[n]+[r]]),s, axis=0) for n,s,r in zip(tile_names,tile_shapes,regions)]))
        
        y_train, y_test, y_val = y_part
    
    else:
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
