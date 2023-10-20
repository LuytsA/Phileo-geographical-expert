import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader

import buteo as beo
import numpy as np
from datetime import date

import sys; sys.path.append("./")

from models.model_CoreCNN_versions import Core_base, Core_tiny, Core_femto, Core_nano
from models.model_Mixer_versions import Mixer_base, Mixer_pico, Mixer_femto, Mixer_nano

import os
from glob import glob
from tqdm import tqdm
import json
import random

import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import config_geography
import cartopy.crs as crs
import cartopy.feature as cfeature
from datetime import datetime 

from utils.visualisations import visualise, render_s2_as_rgb, clip_kg_map, show_tiled, patches_to_array
from utils.encoding_utils import decode_coordinates, encode_coordinates, decode_date
from utils import load_data
from utils.MvMF_utils import MvMF_visuals

import config_geography
pos_feature_pred = config_geography.feature_positions_predictions
pos_feature_label = config_geography.feature_positions_label

torch.set_float32_matmul_precision('high')


def visualise_prediction_tile(tiles, model,results_dir, vis_coords, data_folder = '/home/andreas/vscode/GeoSpatial/phi-lab-rd/data/road_segmentation/images', tile_size=128):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    kg_colour_map ={k:config_geography.kg_map[k]['colour_code'] for k in config_geography.kg_map.keys()}

    for p in tiles:
        print(p)

        kg = clip_kg_map('/phileo_data/aux_data/beck_kg_map_masked.tif',f'{data_folder}/{p}_s2.tif')
        kg_arr = beo.raster_to_array(kg).astype(np.float32)
        s2_arr = beo.raster_to_array(f'{data_folder}/{p}_s2.tif')[:,:,[0,1,2,3,4,5,6,7,8,9,10]].astype(np.float32)
        
        # # you could try to save these arrays to get the testing to run faster
        # # np.save(f'data_val/{p}_kg.npy',kg_arr)
        # # np.save(f'data_val/{p}_s2.npy',s2_arr)
        # kg_arr = np.load(f'data_val/{p}_kg.npy')
        # s2_arr = np.load(f'data_val/{p}_s2.npy')

        region = p
        coord_bbox = beo.raster_to_metadata(f'{data_folder}/{p}_s2.tif')['bbox_latlng']

        patches = beo.array_to_patches(s2_arr,tile_size=tile_size,n_offsets=0)
        total_patches = len(patches)
        
        s2_nodata =np.isin(patches[:,:,:,0],[0,1]) # band 0 contains the SCL classes, [0,1] are defective or no-data pixels
        im_poi = np.where(np.mean(s2_nodata, axis=(1,2))<0.5)[0]
        patches = patches[im_poi][:,:,:,1:]

        np.divide(patches, 10000.0, out=patches)
        patches = beo.channel_last_to_first(patches)

        (mission_id, prod_level, datatake_time,proc_base_number, relative_orbit, tile_number, prod_discriminator) = p.split('/')[-1].split('_')
        datetime_tile = datetime.strptime(datatake_time, "%Y%m%dT%H%M%S")
        first_day = datetime(datetime_tile.year,month=1,day=1)
        day_of_year = (datetime_tile-first_day).days


        dl = DataLoader(patches, batch_size=16, shuffle=False, num_workers=0, drop_last=False, generator=torch.Generator(device=device))


        coords_preds = []
        date_preds = []
        kg_preds = []
        with torch.no_grad():
            for inputs in tqdm(dl):
                batch_pred = model(inputs.to(device))

                y_p = batch_pred.detach().cpu().numpy()

                date_pred = y_p[:,pos_feature_pred['date']]
                coord_pred = y_p[:,pos_feature_pred['coords']]
                kg_pred = y_p[:,pos_feature_pred['kg']].argmax(axis=1)

                date_pred = np.array([decode_date(day) for day in date_pred])

                coords_preds.append(coord_pred)
                date_preds.append(date_pred)
                kg_preds.append(kg_pred)
        date_preds, coords_preds, kg_preds = np.concatenate(date_preds),np.concatenate(coords_preds), np.concatenate(kg_preds)

        # plot figure that show day of year, coordinate and climate zone prediction
        rows,columns =3,2
        fig = plt.figure(figsize=(8 * columns, 8 * rows))

        # rgb of input tile
        fig.add_subplot(rows, columns, 1)
        rgb_image = render_s2_as_rgb(s2_arr[:,:,1:], channel_first=False)
        plt.imshow(rgb_image)
        plt.title(f'{region}')

        # histogram of day of year prediction on the patches
        fig.add_subplot(rows, columns, 2)
        plt.hist([int(pred) for pred in date_preds.flatten()])
        plt.title(f'Predicted day of year on patches DoY={day_of_year}')

        # heatmap of geolocalisation
        fig.add_subplot(rows, columns, 5)
        global_activations, pred_coord, counts = vis_coords.distribution_global_pred(coords_preds)
        ax = fig.add_subplot(rows, columns, (3,4), projection=crs.Robinson())
        vis_coords.plot_globe_dist(ax=ax, dss=global_activations, pred_coord=pred_coord, coord_true = [coord_bbox[0]]+[coord_bbox[2]], counts = counts, linthresh = global_activations.max())
        cmap = matplotlib.colors.ListedColormap([np.array([1.0,1.0,1.0])] + [np.array(v)/255 for v in kg_colour_map.values()], N=32)
        plt.imshow(kg_arr,cmap=cmap, interpolation='nearest',vmin=0,vmax=31)

        # Climate zone prediction for each patch in the tile
        fig.add_subplot(rows, columns, 6)
        kg_preds_reconstructed = -np.ones(shape=(total_patches,))
        kg_preds_reconstructed[im_poi] = kg_preds
        kg_pred_reshaped = patches_to_array(kg_preds_reconstructed, reference=kg_arr, tile_size=1)
        plt.imshow(kg_pred_reshaped,cmap=cmap, interpolation='nearest',vmin=0,vmax=30)

        fig.tight_layout()
        plt.savefig(f'{results_dir}/tile_pred_{tile_number}_{day_of_year}_probdist.png')
        plt.close('all')




def evaluate_model(model, dataloader_test, device, result_dir, num_visualisations = 20):
    num_classes_region=len(config_geography.regions.keys())
    num_classes_kg=len(config_geography.kg_map.keys())
    torch.set_default_device(device)
    model.to(device)
    model.eval()
    
    confmat_region = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes_region).to('cpu')
    confmat_kg = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes_kg).to('cpu')
    running_conf_region = np.zeros((num_classes_region,num_classes_region))
    running_conf_kg = np.zeros((num_classes_kg,num_classes_kg))
    y_pred = []
    x_true = []
    y_true = []
    mse_coords = 0
    if num_visualisations > len(dataloader_test):
        num_visualisations = len(dataloader_test)
    with torch.no_grad():
        for inputs,targets in tqdm(dataloader_test):
            targets = torch.squeeze(targets)
            batch_pred = model(inputs.to(device))

            y_p = batch_pred.detach().cpu()
            y_t = targets.detach().cpu()

            region_pred,region = y_p[:,pos_feature_pred['region']].argmax(axis=1), y_t[:,pos_feature_label['post_aug']['region']]
            coord_pred, coord_true = y_p[:,pos_feature_pred['coords']], y_t[:,pos_feature_label['post_aug']['coords']]
            kg_pred_logits,kg_true = y_p[:,pos_feature_pred['kg']].argmax(axis=1), y_t[:,pos_feature_label['post_aug']['kg']].argmax(axis=1)

            #print(region.shape,region_pred.shape)
            
            #print([(decode_coordinates(cp)-decode_coordinates(ct))**2 for cp,ct in zip(coord_pred,coord_true)])
            ss = np.array([(decode_coordinates(cp)-decode_coordinates(ct))**2 for cp,ct in zip(coord_pred,coord_true)])
            s = np.sum(ss,axis=0)
            # print(ss.shape, s.shape)
            mse_coords += s
            running_conf_region += confmat_region(region_pred,region.squeeze()).numpy()
            running_conf_kg += confmat_kg(kg_pred_logits,kg_true.squeeze()).numpy()

            if len(x_true)<num_visualisations:
                x_true.append(inputs[:1, 0:3, :, :].detach().cpu().numpy()) # only a few per batch to avoid memory issues
                y_pred.append(y_p.numpy())
                y_true.append(y_t.numpy())
    
    y_pred = np.concatenate(y_pred,axis=0)
    x_true = np.concatenate(x_true,axis=0)
    y_true = np.concatenate(y_true,axis=0)

    total = np.sum(running_conf_region)

    s = np.sum(running_conf_region, axis=1, keepdims=True)
    s[s==0]=1
    running_conf_region = running_conf_region/s
    
    s = np.sum(running_conf_kg, axis=1, keepdims=True)
    s[s==0]=1
    running_conf_kg = running_conf_kg/s

    save_path_visualisations = f"{result_dir}/vis.png"
    plt.figure(figsize = (12,9))
    ax = sn.heatmap(running_conf_region, annot=True, fmt='.2f')
    ax.xaxis.set_ticklabels(config_geography.region_inv.values(),rotation = 90)
    ax.yaxis.set_ticklabels(config_geography.region_inv.values(),rotation = 0)
    plt.savefig(save_path_visualisations.replace('vis','cm_region'))
    plt.close()
    
    plt.figure(figsize = (24,18))
    ax = sn.heatmap(running_conf_kg, annot=True, fmt='.2f')
    ax.xaxis.set_ticklabels([config_geography.kg_map[i]['climate_class_str'] for i in config_geography.kg_map.keys()],rotation = 90)
    ax.yaxis.set_ticklabels([config_geography.kg_map[i]['climate_class_str'] for i in config_geography.kg_map.keys()],rotation = 0)
    plt.savefig(save_path_visualisations.replace('vis','cm_kg'))
    plt.close()
    
    batch_size = dataloader_test.batch_size
    metrics = {'acc_regions':running_conf_region.trace()/np.sum(running_conf_region), 'acc_kg':running_conf_kg.trace()/np.sum(running_conf_kg),
               'rmse_lat':np.sqrt(mse_coords[0]/(batch_size*len(dataloader_test))),
               'rmse_long':np.sqrt(mse_coords[1]/(batch_size*len(dataloader_test))),'total_regions':total
               }

    visualise(x_true, np.squeeze(y_true), np.squeeze(y_pred), images=num_visualisations, channel_first=True, vmin=0, vmax=0.5, save_path=save_path_visualisations)
    print(metrics)
    return metrics
        


if __name__ == "__main__":

    DATA_FOLDER = '/phileo_data/mini_foundation/mini_foundation_tifs'

    # model specifics
    model_dir = 'trained_models/12102023_CoreEncoder_LEO_geoMvMF_augm'
    model_name = 'CoreEncoder'
    version ='last' #best
    epoch = '19'

    best_centers = np.load(f'{model_dir}/{model_name}_{version}_centers_{epoch}.npy')
    best_densities = np.load(f'{model_dir}/{model_name}_{version}_densities_{epoch}.npy')
    vis_coords = MvMF_visuals(centers=best_centers, densities=best_densities, encode_centers=True)

    assert len(pos_feature_pred['coords'])==len(best_centers), f"Number of centers in config_geography.py does not match number of centers given ({len(pos_feature_pred['coords'])} vs {len(best_centers)})"
    # load model
    if model_name =='Mixer':
        model = Mixer_nano(chw = (10,128,128), output_dim=31 + len(best_centers) + 2, )
    if model_name =='CoreEncoder':
        model = Core_nano(input_dim=10, output_dim=31 + len(best_centers) + 2)    # model = torch.compile(model)
    
    best_sd = torch.load(os.path.join(model_dir, f"{model_name}_{version}_{epoch}.pt"))
    model.load_state_dict(best_sd)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # get validation files
    # tiles = ['10_points_filtered_22_07/S2A_MSIL2A_20220727T134711_N0400_R024_T21KXA_20220727T213102','10_points_filtered_22_04/S2B_MSIL2A_20220413T221939_N0400_R029_T60KYG_20220414T002214', '10_points_filtered_22_04/S2B_MSIL2A_20220429T023239_N0400_R103_T50JLN_20220429T060411','10_points_filtered_22_10/S2B_MSIL2A_20221020T183409_N0400_R027_T11TPL_20221020T222333'] 
    x_train_files = sorted(glob(os.path.join('/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/', f"*/*train_s2.npy")))
    random.Random(12345).shuffle(x_train_files)
    x_val_files = x_train_files[:20] # validation files as chosen in train_script.py
    tiles =[f.split('.npy')[0].split('patches_labeled/')[-1].split('_train_s2')[0] for f in x_val_files]

    # make folder to store results
    results_dir = f'{model_dir}/results'
    os.makedirs(results_dir, exist_ok=True)

    # visuals on entire tile, quite slow :'(
    visualise_prediction_tile(tiles,model=model,data_folder=DATA_FOLDER, results_dir=results_dir, vis_coords=vis_coords)

    # # This might be broken at this point,not sure
    # evaluate_model(model, dataloader_test, device, result_dir, num_visualisations = 20):