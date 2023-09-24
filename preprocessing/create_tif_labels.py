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
from datetime import datetime
import pandas as pd
from functools import partial
import concurrent.futures
import re

def raster_clip_to_reference(
        global_raster:str,
        reference_raster:str,
    ):

    bbox_ltlng = bo.raster_to_metadata(reference_raster)['bbox_latlng']
    bbox_vector = bo.vector_from_bbox(bbox_ltlng, projection=global_raster)
    bbox_vector_buffered = bo.vector_buffer(bbox_vector, distance=0.1)
    global_clipped = bo.raster_clip(global_raster, bbox_vector_buffered, to_extent=True, adjust_bbox=False)

    return global_clipped
                
def create_label_terrain(tile_path:str, 
                    dem_map:str,
                    folder_dst:str,
                    folder_src:str,
                    overwrite:bool=False,
                    dataset:str='downstream'):

    if dataset=='leonardo':
        tile = tile_path.split('_s2.tif')[0].split('/')[-1]
        reference_raster = tile_path #f"{folder_src}/{tile}_0.tif"
    
    elif dataset=='downstream':
        file_name = tile_path.split('/')[-1]
        tile = re.split('_[0-9]+.tif',file_name)[0]
        reference_raster = tile_path #f"{folder_src}/{tile}_0.tif"
    #reference_raster = f"{folder_src}/{tile}_0.tif"

    try:
        if os.path.exists(reference_raster):
            output_path = f'{folder_dst}/{tile}_label_kg.tif'
            if not os.path.exists(output_path) or overwrite:
                # bbox_ltlng = bo.raster_to_metadata(reference_raster)['bbox_latlng']
                # bbox_vector = bo.vector_from_bbox(bbox_ltlng, projection=global_map)
                # bbox_vector_buffered = bo.vector_buffer(bbox_vector, distance=0.1)

                terrain_clipped = raster_clip_to_reference(dem_map, reference_raster=reference_raster)#bo.raster_clip(global_map, bbox_vector_buffered, to_extent=True, adjust_bbox=False)
                terrain_encoded = bo.raster_dem_to_orientation(terrain_clipped, output_raster=output_path,  height_normalisation=True, include_height=True,)
                bo.raster_align(terrain_encoded, out_path=output_path ,reference=reference_raster,method='reference', resample_alg='bilinear', creation_options=["COMPRESS=DEFLATE", "PREDICTOR=3"])
            
            else:
                print(f"{output_path} already exists")
        else:
            print(f"WARNING: {reference_raster} not found")
    except Exception as e:
        print(f'WARNING: tile {tile} failed with error: \n',e)


def create_label_from_global_map(tile_path:str, 
                    global_map:str,
                    folder_dst:str,
                    folder_src:str,
                    resample:str = 'nearest',
                    overwrite:bool=False,
                    dataset:str='downstream'):

    if dataset=='leonardo':
        tile = tile_path.split('_s2.tif')[0].split('/')[-1]
        reference_raster = tile_path #f"{folder_src}/{tile}_0.tif"
    
    elif dataset=='downstream':
        file_name = tile_path.split('/')[-1]
        tile = re.split('_[0-9]+.tif',file_name)[0]
        reference_raster = tile_path #f"{folder_src}/{tile}_0.tif"
    #reference_raster = f"{folder_src}/{tile}_0.tif"

    try:
        if os.path.exists(reference_raster):
            output_path = f'{folder_dst}/{tile}_label_kg.tif'
            if not os.path.exists(output_path) or overwrite:
                # bbox_ltlng = bo.raster_to_metadata(reference_raster)['bbox_latlng']
                # bbox_vector = bo.vector_from_bbox(bbox_ltlng, projection=global_map)
                # bbox_vector_buffered = bo.vector_buffer(bbox_vector, distance=0.1)

                global_clipped = raster_clip_to_reference(global_map, reference_raster=reference_raster)#bo.raster_clip(global_map, bbox_vector_buffered, to_extent=True, adjust_bbox=False)
                bo.raster_align(global_clipped, out_path=output_path ,reference=reference_raster,method='reference', resample_alg=resample)
            
            else:
                print(f"{output_path} already exists")
        else:
            print(f"WARNING: {reference_raster} not found")
    except Exception as e:
        print(f'WARNING: tile {tile} failed with error: \n',e)


def create_label_encoded_coordinates(tile_path:str, 
                    encoded_coords_map:str,
                    folder_dst:str,
                    folder_src:str,
                    overwrite:bool=True,
                    dataset:str='downstream'):
    
    assert dataset in ['downstream','leonardo'], f"dataset must be either 'downstream' or 'leonardo'"
    
    if dataset=='leonardo':
        tile = tile_path.split('_s2.tif')[0].split('/')[-1]
        reference_raster = tile_path #f"{folder_src}/{tile}_0.tif"
        (mission_id, prod_level, datatake_time,proc_base_number, relative_orbit, tile_number, prod_discriminator) = tile.split('_')
        datetime_tile = datetime.strptime(datatake_time, "%Y%m%dT%H%M%S")
        first_day = datetime(datetime_tile.year,month=1,day=1)
        day_of_year = (datetime_tile-first_day).days
        day_encoded = ((1+np.sin(2*np.pi*day_of_year/365))/2, (1+np.cos(2*np.pi*day_of_year/365))/2)
        #output_path = f'{folder_dst}/{tile}_label_spacetime.tif'

    elif dataset=='downstream':
        file_name = tile_path.split('/')[-1]
        tile = re.split('_[0-9]+.tif',file_name)[0]
        reference_raster = tile_path
    
    output_path = f'{folder_dst}/{tile}_label_coords.tif'


    try:
        if os.path.exists(reference_raster):
            if not os.path.exists(output_path) or overwrite:
                # bbox_ltlng = bo.raster_to_metadata(reference_raster)['bbox_latlng']
                # bbox_vector = bo.vector_from_bbox(bbox_ltlng, projection=encoded_coords_map)
                # bbox_vector_buffered = bo.vector_buffer(bbox_vector, distance=0.1)

                # raster_clipped = bo.raster_clip(encoded_coords_map, bbox_vector_buffered, to_extent=True, adjust_bbox=False)
                raster_clipped = raster_clip_to_reference(encoded_coords_map, reference_raster=reference_raster)
                if dataset=='leonardo':
                    coord_ar = bo.raster_to_array(raster_clipped)
                    day_encoded_ar = np.resize(day_encoded,(coord_ar.shape[0],coord_ar.shape[1],2))
                    time_raster = bo.array_to_raster(day_encoded_ar,reference=raster_clipped)
                    bo.raster_align(time_raster, out_path=output_path.replace('_label_coords','_label_time') ,reference=reference_raster,method='reference', creation_options=["COMPRESS=DEFLATE", "PREDICTOR=3"])

                bo.raster_align(raster_clipped, out_path=output_path ,reference=reference_raster,method='reference', resample_alg='bilinear', creation_options=["COMPRESS=DEFLATE", "PREDICTOR=3"])
            
            else:
                print(f"{output_path} already exists")
        else:
            print(f"WARNING: {reference_raster} not found")

    except Exception as e:
        print(f'WARNING: tile {tile} failed with error: \n',e)




def create_labels(folder_dst:str,
                  folder_src:str,
                  tile_paths:list,
                  labels: list = ['label_encoded_coordinates','label_kg'],
                  kg_map = None,
                  encoded_coords_map = None,
                  dem_map = None,
                  num_workers = None,
                  dataset:str='downstream'
    ):
    '''
    For each s2-tile create a tif raster with label.
    label_encoded_coordinates: scaled latitute and encoded longitude (sine and cosine). For each tile the corresponding raster is created from the global encoded_coords_map
    label_kg: Koppen-Geiger climate zones. For each tile the corresponding raster is created from the global kg_map
    label_terrain: Copdem30 

    '''
    
    for label in labels:
    # for tile_p in tqdm(tile_paths):
    #         create_label_encoded_coordinates(tile_path=tile_p, encoded_coords_map = encoded_coords_map, folder_dst = folder_dst, folder_src = folder_src, dataset=dataset)
        if label=='label_encoded_coordinates':
            print(f'creating {label}')
            proc = partial(create_label_encoded_coordinates, encoded_coords_map = encoded_coords_map, folder_dst = folder_dst, folder_src = folder_src, dataset=dataset)


        elif label=='label_kg':
            print(f'creating {label}')
            proc = partial(create_label_from_global_map, folder_dst = folder_dst, folder_src = folder_src, resample='nearest', global_map=kg_map, dataset=dataset)
        
        elif label=='label_terrain':
            print(f'creating {label}')
            proc = partial(create_label_terrain, folder_dst = folder_dst, folder_src = folder_src, global_map=dem_map, dataset=dataset)
        
        else:
            raise ValueError("Only possible labels to be created are [label_encoded_coordinates, label_kg, label_terrain]")
        
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(proc, tile_paths), total=len(tile_paths)))


def main():
    src_folder = '/phileo_data/mini_foundation/mini_foundation_tifs/10_points_filtered_22_10' #'/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/data_testing/10_points_filtered_22_07'  #'/phileo_data/downstream_dataset_raw'# '/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/data_testing/10_points_filtered_22_10' ##'/archive/road_segmentation/images'
    dst_folder = src_folder #src_folder#'/phileo_data/foundation_data/tif_files' #'/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/data_extra_labels' #f'/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/data_testing'
    
    koppen_geiger_map = '/phileo_data/aux_data/beck_kg_map_masked.tif'
    encoded_coords_map =  '/phileo_data/aux_data/encoded_coordinates_global.tif'
    dem_map = '/phileo_data/aux_data/copdem30.vrt'
        

    labels = ['label_encoded_coordinates'] # 'label_kg', 'label_terrain'
    dataset = 'leonardo' # either "downstream" or "leonardo"
     
    # with open(f'/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/train_test_location_all.json', 'r') as f:
    #     train_test_locations = json.load(f)
    
    # tiles = train_test_locations['train_locations'] + train_test_locations['test_locations']
    tile_paths = glob.glob(f"{src_folder}/*_0.tif") if dataset=='downstream' else glob.glob(f"{src_folder}/*_s2.tif")

    create_labels(
        folder_src=src_folder,
        folder_dst=dst_folder,
        tile_paths=tile_paths,
        labels=labels,
        kg_map=koppen_geiger_map,
        encoded_coords_map = encoded_coords_map,
        dem_map = dem_map,
        dataset=dataset,
        num_workers=4
        )
        
if __name__ == '__main__':
    main()