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



def create_label_kg(tile:str, 
                    kg_map:str,
                    folder_dst:str,
                    folder_src:str):

    reference_raster = f"{folder_src}/{tile}_0.tif"

    try:
        if os.path.exists(reference_raster):
            output_path = f'{folder_dst}/{tile}_label_kg.tif'
            if not os.path.exists(output_path):
                bbox_ltlng = bo.raster_to_metadata(reference_raster)['bbox_latlng']
                bbox_vector = bo.vector_from_bbox(bbox_ltlng, projection=kg_map)
                bbox_vector_buffered = bo.vector_buffer(bbox_vector, distance=0.1)

                kg_clipped = bo.raster_clip(kg_map, bbox_vector_buffered, to_extent=True, adjust_bbox=False)
                bo.raster_align(kg_clipped, out_path=output_path ,reference=reference_raster,method='reference', resample_alg='nearest')
            
            else:
                print(f"{output_path} already exists")
        else:
            print(f"WARNING: {reference_raster} not found")
    except Exception as e:
        print(f'WARNING: tile {tile} failed with error: \n',e)


def create_label_encoded_coordinates(tile:str, 
                    encoded_coords_map:str,
                    folder_dst:str,
                    folder_src:str,
                    overwrite=False):
    
    reference_raster = f"{folder_src}/{tile}_0.tif"

    try:
        if os.path.exists(reference_raster):
            output_path = f'{folder_dst}/{tile}_label_coords.tif'
            if not os.path.exists(output_path) or overwrite:
                bbox_ltlng = bo.raster_to_metadata(reference_raster)['bbox_latlng']
                bbox_vector = bo.vector_from_bbox(bbox_ltlng, projection=encoded_coords_map)
                bbox_vector_buffered = bo.vector_buffer(bbox_vector, distance=0.1)

                coords_clipped = bo.raster_clip(encoded_coords_map, bbox_vector_buffered, to_extent=True, adjust_bbox=False)
                bo.raster_align(coords_clipped, out_path=output_path ,reference=reference_raster,method='reference', resample_alg='bilinear')
            
            else:
                print(f"{output_path} already exists")
        else:
            print(f"WARNING: {reference_raster} not found")
    
    except Exception as e:
        print(f'WARNING: tile {tile} failed with error: \n',e)




def create_labels(folder_dst:str,
                  folder_src:str,
                  tiles:list,
                  labels: list = ['label_encoded_coordinates','label_kg'],
                  kg_map = None,
                  encoded_coords_map = None,
                  num_workers = None):
    '''
    For each tile create a tif raster with label.
    label_encoded_coordinates: scaled latitute and encoded longitude (sine and cosine). For each tile the corresponding raster is created from the global encoded_coords_map
    label_kg: Koppen-Geiger climate zones. For each tile the corresponding raster is created from the global kg_map

    '''
    
    for label in labels:
        if label=='label_encoded_coordinates':
            print(f'creating {labels}')
            proc = partial(create_label_encoded_coordinates, encoded_coords_map = encoded_coords_map, folder_dst = folder_dst, folder_src = folder_src)


        elif label=='label_kg':
            print(f'creating {labels}')
            proc = partial(create_label_kg, folder_dst = folder_dst, folder_src = folder_src, kg_map=kg_map)
        
        else:
            raise ValueError("Only possible labels to be created are [label_encoded_coordinates, label_kg]")
        
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(proc, tiles), total=len(tiles)))


def main():
    src_folder = '/phileo_data/downstream_dataset_raw' #'/archive/road_segmentation/images'
    dst_folder = '/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/data_extra_labels' #f'/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/data_testing'
    
    koppen_geiger_map = '/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/beck_kg_map_masked.tif'
    encoded_coords_map =  '/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/encoded_coords.tif'
        

    labels = ['label_kg','label_encoded_coordinates']

    
    with open(f'train_test_location_all.json', 'r') as f:
        train_test_locations = json.load(f)
    
    tiles = train_test_locations['train_locations'] + train_test_locations['test_locations']


    create_labels(
        folder_src=src_folder,
        folder_dst=dst_folder,
        tiles=tiles,
        labels=labels,
        kg_map=koppen_geiger_map,
        encoded_coords_map = encoded_coords_map,
        num_workers=6
        )
        
if __name__ == '__main__':
    main()