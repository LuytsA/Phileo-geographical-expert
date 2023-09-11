import os
import sys; sys.path.append("./")
import random
import csv
import numpy as np
from glob import glob
from tqdm import tqdm
import glob 
import buteo as bo
from matplotlib import pyplot as plt
import json
from datetime import date
import pandas as pd
from functools import partial
import concurrent.futures
import config_geography

LABELS = ['label_roads','label_geography','label_building','label_lc']

def select_and_save_patches(
        tile:str,
        label_patches_dict: dict,
        s2_patches: list, 
        dst_folder:str,
        partition: str = 'train', 
        val_split_ratio: float = 0.1,
    ):
    '''
    Takes patches from an image and saves only those that meet certain criteria.

    Parameters
    ----------
    tile : string
        Name of the tile/location e.g east-africa_0


    label_patches : numpy.array
        Array containing all the patches of the labels
    
    s2_patches : numpy.array
        Array containing all the patches of the images

    dst_folder : str,
        Folder where to save the selected patches

    partition : str, ['train','test]
        Is the tile destined for the train or test set.

    val_split_ratio : float
        If partition=='train', this fraction of the data is saved as validation
    
    Returns
    -------
    None
    '''
    

    # order S2 bands: 0-SCL, 1-B02, 2-B03, 3-B04, 4-B08, 5-B05, 6-B06, 7-B07, 8-B8A, 9-B11, 10-B12
    BANDS_TO_KEEP = [1,2,3,4,5,6,7,8,9,10]
    # SLC classes:  0-no_data 1-saturated_or_defective 8-cloud_medium_prob 9-cloud_high_prob
    SCL_CLOUD_BANDS = [0,1,8,9]
    # Independent of threshold clouds, if the cloud cover in the images exceeds a certain value, the patch is not saved.
    MAX_CLOUD_COVER = 0.1

    # record metadata about each tile, more will be added throughout the processing
    metadata = {'tile':tile, 'bands_kept':BANDS_TO_KEEP, 'SCL_cloud_classes':SCL_CLOUD_BANDS,
                'max_allowed_cloud_cover':MAX_CLOUD_COVER, 'partition':partition, 'val_split_ratio':val_split_ratio if partition=='train' else None,}  


    ## Process s2 timeseries
    # every label has a time series of s2 images associated to it. Loop over them and save those without to much cloud cover 
    selected_s2 = []
    if partition=='train':
        selected_s2_val = []

    timeseries_poi = []
    for k in range(len(s2_patches)):

        s2_arr = s2_patches[k]

        print(tile, s2_arr.shape)

        s2_clouds =np.isin(s2_arr[:,:,:,0],SCL_CLOUD_BANDS) # band 0 contains the SCL classes which can indicate the amount of cloud cover
        im_poi = np.where(np.mean(s2_clouds, axis=(1,2))<MAX_CLOUD_COVER)[0]
        timeseries_poi.append(im_poi)

        print(im_poi[:5])
        if len(im_poi)==0:
            continue 

        s2_patches_tmp = s2_arr[im_poi][:,:,:,BANDS_TO_KEEP]

        if partition=='train':
            idx_val = int(s2_patches_tmp.shape[0] * (1 - val_split_ratio))
            selected_s2_val.append(s2_patches_tmp[idx_val:])
            s2_patches_tmp = s2_patches_tmp[:idx_val]

        selected_s2.append(s2_patches_tmp) 

    selected_s2 = np.concatenate(selected_s2)

    # save s2 patches
    if partition=='train':
        selected_s2_val = np.concatenate(selected_s2_val)
    
        np.save(f'{dst_folder}/{tile}_train_s2.npy', selected_s2)
        np.save(f'{dst_folder}/{tile}_val_s2.npy', selected_s2_val)

        len_selected_s2 = selected_s2_val.shape[0]+selected_s2.shape[0]
        selected_shape_s2 = (len_selected_s2,) + selected_s2.shape[1:]

    else:
        np.save(f'{dst_folder}/{tile}_test_s2.npy', selected_s2)
        selected_shape_s2 = selected_s2.shape

        

    ## Process labels
    for label_name, label_patches in label_patches_dict.items():
        label_shape = label_patches.shape
        selected_labels = []
        if partition=='train':
            selected_labels_val = []

        # loop over cloudless patches for each timeseries s2 image and concatenate the labels
        for label_poi in timeseries_poi:
            label_patches_tmp = label_patches[label_poi]

            if partition=='train':
                idx_val = int(label_patches_tmp.shape[0] * (1 - val_split_ratio))
                selected_labels_val.append(label_patches_tmp[idx_val:])
                label_patches_tmp = label_patches_tmp[:idx_val]
            
            selected_labels.append(label_patches_tmp) 

        selected_labels = np.concatenate(selected_labels)
        assert selected_labels.shape[0] == selected_s2.shape[0], f"Number of patches for labels and images do not match for {tile}."

        # save the labels to .np files
        if partition=='train':
            selected_labels_val = np.concatenate(selected_labels_val)
            assert selected_labels_val.shape[0] == selected_s2_val.shape[0],  f"Number of patches for labels and images do not match for {tile}."

            np.save(f'{dst_folder}/{tile}_train_{label_name}.npy', selected_labels)
            np.save(f'{dst_folder}/{tile}_val_{label_name}.npy', selected_labels_val)

            len_selected_label = selected_labels_val.shape[0]+selected_labels.shape[0]
            selected_shape_label = (len_selected_label,) + selected_labels.shape[1:]
        
        else:
            np.save(f'{dst_folder}/{tile}_test_{label_name}.npy', selected_labels)
            selected_shape_label = selected_labels.shape

        metadata[f'{label_name}_shape'] = tuple(selected_shape_label)
        metadata['fraction_images_kept'] = float(selected_shape_s2[0]/(len(timeseries_poi)*label_shape[0]))
    

    # create output folder 
    os.makedirs(f'{dst_folder}/metadata', exist_ok=True)
    
    with open(f'{dst_folder}/metadata/{tile}.json', 'w') as fp:
        json.dump(metadata, fp)



def process_tile(
        tile: str,
        folder_src: str,
        folder_dst: str,
        aux_data: str,
        overlaps: int = 1,
        patch_size: int = 128,
        val_split_ratio: float = 0.1,
        partition: str = 'train',
):

    '''
    Split a tile/location in patches and save the patches satisfying certain criteria.

    Parameters
    ----------
    tile : string
        Name of the tile/location e.g east-africa_0

    folder_src : str,
        Folder where look for the tile. The path to the tile should be {folder_src}/tile**.tiff

    folder_dst : str,
        Folder where to save the selected patches
    
    overlaps: int,
        Offsets in x and y direction of the images. Higher overlap creates more patches from the same image.
        
    patch_size : int,
        x- and y-dimension of the resulting patches

    val_split_ratio : float
        If partition=='train', this fraction of the data is saved as validation
    
    partition : str, ['train','test]
        Is the tile destined for the train or test set.
    Returns
    -------
    None
    '''

    timeseries_s2_raster = glob.glob(f'{folder_src}/{tile}_*[0-9].tif')
    labels_raster = {}
    for label in LABELS:
        label_path = f'{folder_src}/{tile}_{label}.tif'
        if os.path.exists(label_path):
            labels_raster[label] = label_path
        
        # temporary, due to naming mismatch
        if os.path.exists(f'{folder_src}/{tile}_0_{label}.tif'):
            labels_raster[label] = f'{folder_src}/{tile}_0_{label}.tif'

    try:
        assert len(labels_raster.keys())>0, f'no labels found for tile {tile}'
        assert bo.check_rasters_are_aligned(timeseries_s2_raster+list(labels_raster.values())), 'labels and images are not aligned'

        labels_arr = {label:bo.raster_to_array(label_path) for label, label_path in labels_raster.items()}
        labels_patches = {label:bo.array_to_patches(arr, tile_size=patch_size, n_offsets=overlaps) for label, arr in labels_arr.items()}


        # turn each image of timeseries to patches
        timeseries_s2_arr= [bo.raster_to_array(s2_t) for s2_t in timeseries_s2_raster]
        timeseries_s2_patches= [bo.array_to_patches(arr,tile_size=patch_size, n_offsets=overlaps) for arr in timeseries_s2_arr]

        select_and_save_patches(tile=tile, label_patches_dict=labels_patches, s2_patches=timeseries_s2_patches, dst_folder=folder_dst,
                    partition=partition, val_split_ratio=val_split_ratio)

    except Exception as e:
        print(f'WARNING: tile {tile} failed with error: \n',e)


def process_data(
        folder_src: str,
        folder_dst: str,
        aux_data: str,
        overlaps: int = 1,
        patch_size: int = 128,
        val_split_ratio: float = 0.1,
        test_locations: list = None,
        train_locations: list = None,
        num_workers: int = None
):
    
    '''
    Process a set of .tiff files to numpy arrays. Every tiff will results in a numpy array of dimension patches x patch_size x patch_size x channels

    Parameters
    ----------
    folder_src : str,
        Folder where tiff tiles are located. The path to the tile should be {folder_src}/tile**.tiff

    folder_dst : str,
        Folder where to save the selected patches
    
    overlaps: int,
        Offsets in x and y direction of the images. Higher overlap creates more patches from the same image.
        
    patch_size : int,
        x- and y-dimension of the resulting patches

    val_split_ratio : float
        If partition=='train', this fraction of the data is saved as validation
    
    test_locations : List[str]
        List of the location that should be in the test set e.g. ['east-africa_0','north-america_5']
    
    train_locations : List[str]
        List of the location that should be in the train set e.g. ['east-africa_0','north-america_5']
        From this set val_split_ratio will be save as validation set.

    num_workers: int
        Max number of workers to process different tiles in parallel

    Returns
    -------
    None
    '''
    
    if train_locations is None:
        # If empty, all locations are used.
        train_locations = []

    if test_locations is None:
        test_locations = []

    for x in train_locations:
        if x in test_locations:
            raise ValueError("Location in both train and test.")


    print('processing train locations ...')
    proc = partial(process_tile, folder_dst = folder_dst, folder_src = folder_src, aux_data=aux_data,overlaps=overlaps, patch_size=patch_size, val_split_ratio=val_split_ratio, partition='train')
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(proc, train_locations), total=len(train_locations)))

    print('processing test locations ...')
    proc = partial(process_tile, folder_dst = folder_dst, folder_src = folder_src, aux_data=aux_data, overlaps=overlaps, patch_size=patch_size, val_split_ratio=val_split_ratio, partition='test')
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(proc, test_locations), total=len(test_locations)))


    # merge all metadata created
    metadata = glob.glob(f'{folder_dst}/metadata/**.json')
    metadata_merged = {}
    for f in metadata:
        with open(f, 'r') as p:
            m = json.load(p) 
            tile = m.pop('tile',f)
            metadata_merged[tile] = m
        os.remove(f)
    with open(os.path.join(folder_dst, 'metadata', 'tiles_metadata.json'), 'w') as fp:
        json.dump(metadata_merged, fp)

    # create csv for datasplitting protocol
    files = glob.glob(os.path.join(folder_dst, '**_train_s2.npy'))
    info = pd.DataFrame()
    info['samples'] = []
    for f in files:
        arr = np.load(f,mmap_mode='r')
        info.loc[os.path.relpath(f,folder_dst)] = arr.shape[0]
    file_name = f'{folder_dst}/{date.today().strftime("%d%m%Y")}_npy_file_info.csv'
    info.to_csv(file_name)



def main():
    src_folder = 'data_testing' #'/archive/road_segmentation/images'
    dst_folder = f'data_geography/'
    aux_data = '/phileo/aux_data'
    # koppen_geiger_map = '/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/beck_kg_map_masked.tif'
        
    N_OFFSETS = 0
    PATCH_SIZE = 64
    VAL_SPLIT_RATIO = 0.1
    
    # with open(f'{aux_data}/train_test_locations.json', 'r') as f:
    #     train_test_locations = json.load(f)


    train_locations = ['isreal-2_3']#['denmark-1_1','east-africa_2','isreal-2_3','isreal-2_5','east-africa_3']#train_test_locations['train_locations']
    test_locations = [] #train_test_locations['test_locations']

    process_data(
        folder_src=src_folder,
        folder_dst=dst_folder,
        aux_data=aux_data,
        overlaps=N_OFFSETS,
        patch_size=PATCH_SIZE,
        val_split_ratio=VAL_SPLIT_RATIO,
        test_locations=test_locations,
        train_locations=train_locations,
        num_workers=1
)
        

if __name__ == '__main__':
    main()


