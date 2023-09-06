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

def kg_patch_to_label(patches,tile):
    # count occurence of each label. Final label array consists of the weighted occurences for each label
    label =np.zeros((patches.shape[0],len(config_geography.kg_map.keys())))
    for i in range(patches.shape[0]):
        u = np.unique(patches[i], return_counts=True,)
        label[i,u[0]] = u[1]/np.sum(u[1])

    # add region and coordinates info (same for every patch in the tile)
    with open('/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/tiles_coordinates.json') as fp:
        tiles_coord = json.load(fp)
    coord = tiles_coord[tile]
    region = config_geography.regions[tile.split('_')[0]]
    label = np.concatenate([label,np.repeat(np.array([coord+[region]]),label.shape[0], axis=0)], axis=-1)

    return label

def clip_kg_map(
        kg:str,
        reference: str
    ):

    bbox_ltlng = bo.raster_to_metadata(reference)['bbox_latlng']
    bbox_vector = bo.vector_from_bbox(bbox_ltlng, projection=kg)
    bbox_vector_buffered = bo.vector_buffer(bbox_vector, distance=0.1)

    kg_clipped = bo.raster_clip(kg, bbox_vector_buffered, to_extent=True, adjust_bbox=False)


    kg_aligned = bo.raster_align(kg_clipped,reference=reference,method='reference', resample_alg='nearest')
    
    bo.delete_dataset_if_in_memory_list([kg_clipped,bbox_vector,bbox_vector_buffered])

    return kg_aligned


def select_and_save_patches(
        tile:str,
        label_patches: np.array,
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

    label_shape = label_patches.shape

    # # handle any potential errors
    # np.clip(label_patches, 0.0, 1.0, out=label_patches)
    # label_patches[np.isnan(label_patches)] = 0.0

    # every label has 3 s2 images associated to it. Loop over them and save those without to much cloud cover 
    selected_labels = []
    selected_s2 = []
    if partition=='train':
        selected_s2_val = []
        selected_labels_val = []

    for k in range(3):
        s2_arr = s2_patches[k]

        s2_clouds =np.isin(s2_arr[:,:,:,0],SCL_CLOUD_BANDS) # band 0 contains the SCL classes indicating cloud cover

        im_poi = np.where(np.mean(s2_clouds, axis=(1,2))<MAX_CLOUD_COVER)[0]
        
        label_patches_tmp = label_patches[im_poi]
        s2_patches_tmp = s2_arr[im_poi][:,:,:,BANDS_TO_KEEP]

        if partition=='train':
            idx_val = int(label_patches_tmp.shape[0] * (1 - val_split_ratio))

            selected_labels_val.append(label_patches_tmp[idx_val:])
            selected_s2_val.append(s2_patches_tmp[idx_val:])

            label_patches_tmp = label_patches_tmp[:idx_val]
            s2_patches_tmp = s2_patches_tmp[:idx_val]

        selected_labels.append(label_patches_tmp) 
        selected_s2.append(s2_patches_tmp) 

    selected_labels = np.concatenate(selected_labels)
    selected_s2 = np.concatenate(selected_s2)

    selected_labels = kg_patch_to_label(selected_labels,tile)
    # if partition=='train':
    #     selected_labels_val = kg_patch_to_label(selected_labels_val,tile)


    # # count occurence of each label. Final label array consists of the weighted occurences for each label
    # label_smooth =np.zeros((selected_labels.shape[0],len(config_geography.kg_map.keys())))
    # for i in range(selected_labels.shape[0]):
    #     u = np.unique(selected_labels[i], return_counts=True,)
    #     label_smooth[i,u[0]] = u[1]/np.sum(u[1])
    # selected_labels = label_smooth

    # # add region and coordinates info (same for every patch in the tile)
    # with open('/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/tiles_coordinates.json') as fp:
    #     tiles_coord = json.load(fp)
    # coord = tiles_coord[tile]
    ##region = config_geography.regions[tile.split('_')[0]]
    # selected_labels = np.concatenate([selected_labels,np.repeat(np.array([coord+[region]]),selected_labels.shape[0], axis=0)], axis=-1)
    # if partition=='train':


    
    assert selected_labels.shape[0] == selected_s2.shape[0], "Number of patches do not match."

    # create output folder 
    os.makedirs(f'{dst_folder}/metadata', exist_ok=True)

    if partition=='train':
        selected_labels_val = np.concatenate(selected_labels_val)
        selected_s2_val = np.concatenate(selected_s2_val)
        selected_labels_val = kg_patch_to_label(selected_labels_val,tile)

        assert selected_labels_val.shape[0] == selected_s2_val.shape[0], "Number of patches do not match."
    
        np.save(f'{dst_folder}/{tile}_train_s2.npy', selected_s2)
        np.save(f'{dst_folder}/{tile}_train_label_geography.npy', selected_labels)
        np.save(f'{dst_folder}/{tile}_val_s2.npy', selected_s2_val)
        np.save(f'{dst_folder}/{tile}_val_label_geography.npy', selected_labels_val)

        len_selected_s2 = selected_s2_val.shape[0]+selected_s2.shape[0]
        selected_shape_s2 = (len_selected_s2,) + selected_s2.shape[1:]
        len_selected_label = selected_labels_val.shape[0]+selected_labels.shape[0]
        selected_shape_label = (len_selected_label,) + selected_labels.shape[1:]
    else:
        np.save(f'{dst_folder}/{tile}_test_s2.npy', selected_s2)
        np.save(f'{dst_folder}/{tile}_test_label_geography.npy', selected_labels)
        selected_shape_s2 = selected_s2.shape
        selected_shape_label = selected_labels.shape


    metadata = {'tile':tile, 'image_shape':tuple(selected_shape_s2), 'label_shape':tuple(selected_shape_label), 'label_format': '31 kg_classes + lat + long + region_id',
                'fraction_images_kept':float(selected_shape_s2[0]/(3*label_shape[0])), 'bands_kept':BANDS_TO_KEEP, 'SCL_cloud_classes':SCL_CLOUD_BANDS,
                'max_allowed_cloud_cover':MAX_CLOUD_COVER,
                'partition':partition, 'val_split_ratio':val_split_ratio if partition=='train' else None,
                }  
    
    with open(f'{dst_folder}/metadata/{tile}.json', 'w') as fp:
        json.dump(metadata, fp)



def process_tile(
        tile: str,
        folder_src: str,
        folder_dst: str,
        koppen_geiger_map: str,
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

    im1 = f'{folder_src}/{tile}_0.tif'
    im2 = f'{folder_src}/{tile}_1.tif'
    im3 = f'{folder_src}/{tile}_2.tif'
    label = clip_kg_map(koppen_geiger_map,im1)[0]

    try:
        assert bo.check_rasters_are_aligned([label,im1,im2,im3]), 'labels and images are not aligned'

        label_arr = bo.raster_to_array(label)
        label_patches = bo.array_to_patches(label_arr, tile_size=patch_size, n_offsets=overlaps)


        # save s2 patches (3 images per label)
        im_patches = []
        for i,raster in enumerate([im1,im2,im3]):
            arr = bo.raster_to_array(raster)
            im_patches.append(bo.array_to_patches(arr, tile_size=patch_size, n_offsets=overlaps))

        select_and_save_patches(tile=tile, label_patches=label_patches, s2_patches=im_patches, dst_folder=folder_dst,
                    partition=partition, val_split_ratio=val_split_ratio)

    except Exception as e:
        print(f'WARNING: tile {tile} failed with error: \n',e)


def process_data(
        folder_src: str,
        folder_dst: str,
        koppen_geiger_map:str,
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
    proc = partial(process_tile, folder_dst = folder_dst, folder_src = folder_src, koppen_geiger_map=koppen_geiger_map, overlaps=overlaps, patch_size=patch_size, val_split_ratio=val_split_ratio, partition='train')
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(proc, train_locations), total=len(train_locations)))

    print('processing test locations ...')
    proc = partial(process_tile, folder_dst = folder_dst, folder_src = folder_src, koppen_geiger_map=koppen_geiger_map,overlaps=overlaps, patch_size=patch_size, val_split_ratio=val_split_ratio, partition='test')
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
    # REGIONS =['east-africa', 'europe','eq-guinea', 'japan','south-america', 'north-america', 'nigeria', 'senegal']
    src_folder = '/archive/road_segmentation/images' #'/home/andreas/vscode/GeoSpatial/phi-lab-rd/data/road_segmentation/images'
    dst_folder = f'data_geography/'
    koppen_geiger_map = '/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/Beck_KG_V1_present_0p0083.tif'
        
    n_offsets= 0
    tile_size=64
    val_split_ratio = 0.1
    
    with open('utils/roads_train_test_locations.json', 'r') as f:
        train_test_locations = json.load(f)


    train_locations = train_test_locations['train_locations']
    test_locations = train_test_locations['test_locations']

    process_data(
        folder_src=src_folder,
        folder_dst=dst_folder,
        koppen_geiger_map=koppen_geiger_map,
        overlaps=n_offsets,
        patch_size=tile_size,
        val_split_ratio=val_split_ratio,
        test_locations=test_locations,
        train_locations=train_locations,
        num_workers=6
)
        

if __name__ == '__main__':
    main()
    # dst_folder = '/home/andreas/vscode/GeoSpatial/Phileo-downstream-tasks/data_landcover'
    # tile = 'north-america_4'

    # visualize_labels(dst_folder, tile, plt_num= 16)

