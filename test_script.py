import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader

import buteo as beo
import numpy as np
from datetime import date

import sys; sys.path.append("./")

from models.model_CoreCNN import Core_base, Core_tiny

import os
from glob import glob
from tqdm import tqdm
import json

import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import cv2
import config_geography
import cartopy.crs as crs
import cartopy.feature as cfeature
from datetime import datetime 

from utils import (
    load_data,
    data_protocol_bd,
    visualise,
    render_s2_as_rgb,
    decode_coordinates,
    decode_time
)
from preprocessing.preprocess_geo import clip_kg_map
import config_geography
pos_feature_pred = config_geography.feature_positions_predictions
pos_feature_label = config_geography.feature_positions_label

torch.set_float32_matmul_precision('high')

def precision_recall(y_pred_classification,y_test_classification):
    diff = y_pred_classification.astype(int)-y_test_classification.astype(int)
    fp = np.count_nonzero(diff == 1) # false positives
    fn = np.count_nonzero(diff == -1) # false negatives
    tp = np.sum(y_test_classification) - fn # true positives = all positives - false negatives

    return np.array([tp, fp, fn])

def show_tiled(rgb_array, tile_size, line_thickness=10):
    h,w =rgb_array.shape[0], rgb_array.shape[1]
    h_lines = h//tile_size + 1
    w_lines = w//tile_size + 1

    for l in range(h_lines):
        black = [tile_size*l +i for i in range(int(line_thickness)) ]
        rgb_array[black,:,:] = [0,0,0]
    for l in range(w_lines):
        black = [tile_size*l +i for i in range(int(line_thickness)) ]
        rgb_array[:,black,:] = [0,0,0]
    
    return rgb_array

def patches_to_array(patches, reference, tile_size=64):
    h,w,c = reference.shape
    n_patches = patches.shape[0]
    # Reshape the patches for stitching
    print('patches',patches.shape,'arr',reference.shape)
    reshape = patches[:-171].reshape(
        int(np.sqrt(n_patches))-1,
        int(np.sqrt(n_patches))-1,
        tile_size,
        tile_size,
        1,
        1,
    )

    # Swap axes to rearrange patches in the correct order for stitching
    swap = reshape.swapaxes(1, 2)

    # Combine the patches into a single array
    destination = swap.reshape(
        int(np.sqrt(n_patches))-1,
        int(np.sqrt(n_patches))-1,
        1,
    )

    return destination

def visualise_prediction_tile(tiles, model,results_dir, data_folder = '/home/andreas/vscode/GeoSpatial/phi-lab-rd/data/road_segmentation/images', label='lc', tile_size=128):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    kg_colour_map ={k:config_geography.kg_map[k]['colour_code'] for k in config_geography.kg_map.keys()}

    for p in tiles:
        print(p)
        #try:
        kg = clip_kg_map('/phileo_data/aux_data/beck_kg_map_masked.tif',f'{data_folder}/{p}_s2.tif')[0]
        kg_arr = beo.raster_to_array(kg).astype(np.float32)
        s2_arr = beo.raster_to_array(f'{data_folder}/{p}_s2.tif')[:,:,[0,1,2,3,4,5,6,7,8,9,10]].astype(np.float32)
        #s2_SCL = s2_arr[:,:,0]
        #s2_arr = s2_arr[:,:,1:]
        region = p
        coord_bbox = beo.raster_to_metadata(f'{data_folder}/{p}_s2.tif')['bbox_latlng']

        patches = beo.array_to_patches(s2_arr,tile_size=tile_size,n_offsets=0)
        total_patches = len(patches)
        
        s2_nodata =np.isin(patches[:,:,:,0],[0,1]) # band 0 contains the SCL classes, [0,1] are defective or no-data pixels
        im_poi = np.where(np.mean(s2_nodata, axis=(1,2))<0.5)[0]
        # patches_kg = beo.array_to_patches(kg_arr,tile_size=64,n_offsets=0)
        patches = patches[im_poi][:,:,:,1:]

        np.divide(patches, 10000.0, out=patches)
        patches = beo.channel_last_to_first(patches)

        (mission_id, prod_level, datatake_time,proc_base_number, relative_orbit, tile_number, prod_discriminator) = p.split('/')[-1].split('_')
        datetime_tile = datetime.strptime(datatake_time, "%Y%m%dT%H%M%S")
        first_day = datetime(datetime_tile.year,month=1,day=1)
        day_of_year = (datetime_tile-first_day).days


        # x_tile = beo.MultiArray([patches])
        # y_dummy = beo.MultiArray([np.zeros(patches.shape[0])])
        # dl,_,_ = load_data(x_tile, y_dummy,[0],[0],[0],[0],with_augmentations=False,batch_size=16, encoder_only=True)
        dl = DataLoader(patches, batch_size=16, shuffle=False, num_workers=0, drop_last=False, generator=torch.Generator(device=device))


        coords_preds = []
        date_preds = []
        kg_preds = []
        with torch.no_grad():
            for inputs in tqdm(dl):
                batch_pred = model(inputs.to(device))

                y_p = batch_pred.detach().cpu().numpy()

                date_pred = y_p[:,pos_feature_pred['time']]
                coord_pred = y_p[:,pos_feature_pred['coords']]
                kg_pred = y_p[:,pos_feature_pred['kg']].argmax(axis=1)
            
                coord_pred = np.array([decode_coordinates(co) for co in coord_pred])
                date_pred = np.array([decode_time(day) for day in date_pred])

                coords_preds.append(coord_pred)
                date_preds.append(date_pred)
                kg_preds.append(kg_pred)
        date_preds, coords_preds, kg_preds = np.concatenate(date_preds),np.concatenate(coords_preds), np.concatenate(kg_preds)

        
        rows,columns =3,2
        fig = plt.figure(figsize=(8 * columns, 8 * rows))

        fig.add_subplot(rows, columns, 1)
        rgb_image = render_s2_as_rgb(s2_arr[:,:,1:], channel_first=False)
        plt.imshow(show_tiled(rgb_image, tile_size))
        plt.title(f'{region}')

        fig.add_subplot(rows, columns, 2)
        plt.hist([int(pred) for pred in date_preds.flatten()])
        plt.title(f'Predicted day of year on patches DoY={day_of_year}')
        # plt.savefig(f'region_dist_{p}.png')
        # plt.close()

        print(coord_bbox)
        print(coords_preds[:5,1])
        print(coords_preds[:5,0])

        ax = fig.add_subplot(rows, columns, (3,4), projection=crs.Robinson())
        #ax = fig.add_subplot(1,1,1, projection=crs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.gridlines()
        plt.scatter(x=coords_preds[:,1], y= coords_preds[:,0],  #x=long, y=lat
                    color="green",
                    s=1,
                    alpha=0.5,
                    transform=crs.PlateCarree()) ## Important
        
        sea_patches = np.where(kg_preds==0)
        plt.scatter(x=coords_preds[[sea_patches],1], y= coords_preds[[sea_patches],0],  #x=long, y=lat
                    color="dodgerblue",
                    s=1,
                    alpha=0.5,
                    transform=crs.PlateCarree()) ## Important
        
        plt.scatter(x=coord_bbox[2:], y= coord_bbox[:2], #x=long, y=lat
                    color="red",
                    s=16,
                    transform=crs.PlateCarree()) ## Important
        plt.title('coordinate predictions on patches')

        fig.add_subplot(rows, columns, 5)
        cmap = matplotlib.colors.ListedColormap([np.array([1.0,1.0,1.0])] + [np.array(v)/255 for v in kg_colour_map.values()], N=32)
        plt.imshow(kg_arr,cmap=cmap, interpolation='nearest',vmin=0,vmax=31)

        fig.add_subplot(rows, columns, 6)
        print(kg_preds.shape)
        kg_preds_reconstructed = -np.ones(shape=(total_patches,))
        kg_preds_reconstructed[im_poi] = kg_preds
        kg_pred_reshaped = patches_to_array(kg_preds_reconstructed, reference=kg_arr, tile_size=1)
        plt.imshow(kg_pred_reshaped,cmap=cmap, interpolation='nearest',vmin=0,vmax=30)

        fig.tight_layout()
        plt.savefig(f'{results_dir}/tile_pred_{tile_number}_{day_of_year}.png')
        plt.close('all')
        
        #except Exception as e:
        #    print(e)
        



# def grad_cam_vis(model, img, label):
#         pred = model(img)#.argmax(dim=1)

#         pred_class = pred.argmax(dim=1)
#         pred[:,pred_class].backward()
#         gradients = model.get_activations_gradient()

#         # pool the gradients across the channels
#         pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

#         # get the activations of the last convolutional layer
#         activations = model.get_activations(img).detach()

#         for i in range(768):
#             activations[:, i, :, :] *= pooled_gradients[i]
#         print(gradients.shape, pooled_gradients.shape, activations.shape)

#         # average the channels of the activations
#         heatmap = torch.mean(activations, dim=1).squeeze()

#         # relu on top of the heatmap
#         # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
#         heatmap = np.maximum(heatmap, 0)

#         # normalize the heatmap
#         heatmap /= torch.max(heatmap)

#         # draw the heatmap
#         import matplotlib.pyplot as plt
#         plt.matshow(heatmap.squeeze())
#         # plt.savefig('gradcam.png')

#         # print(img.shape,label.shape)
#         # visualise([img.numpy()[0]], [label.numpy()[0]], y_pred=pred.argmax(dim=1),channel_first=True, images=1, class_dict=None, save_path='vis_grad.png')

#         img = render_s2_as_rgb(img.numpy()[0], channel_first=True)
#         # img = cv2.imread('./data/Elephant/data/05fig34.jpg')
#         heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
#         heatmap = np.uint8(255 * heatmap)
#         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#         superimposed_img = heatmap * 0.4 + img

#         fig = plt.figure(figsize=(10 * 1, 10 * 1))
#         fig.add_subplot(1, 3, 1)
#         plt.imshow(img)
#         fig.add_subplot(1, 3, 2)
#         plt.imshow(heatmap)
#         fig.add_subplot(1, 3, 3)
#         plt.imshow(superimposed_img/255)
#         plt.savefig(f'heatmap_{j}.png')


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
    #y_vis = [i*batch_size for i in range(0,num_visualisations)]
    metrics = {'acc_regions':running_conf_region.trace()/np.sum(running_conf_region), 'acc_kg':running_conf_kg.trace()/np.sum(running_conf_kg),
               'rmse_lat':np.sqrt(mse_coords[0]/(batch_size*len(dataloader_test))),
               'rmse_long':np.sqrt(mse_coords[1]/(batch_size*len(dataloader_test))),'total_regions':total
               }

    #plt.savefig(save_path_visualisations.replace("vis","cm22"))
    visualise(x_true, np.squeeze(y_true), np.squeeze(y_pred), images=num_visualisations, channel_first=True, vmin=0, vmax=0.5, save_path=save_path_visualisations)
    print(metrics)
    return metrics
        


if __name__ == "__main__":

    DATA_FOLDER = '/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/data_geography' #'/home/andreas/vscode/GeoSpatial/Phileo-downstream-tasks/data_landcover'
    REGIONS = ['north-america','east-africa', 'europe','eq-guinea', 'japan','south-america', 'nigeria', 'senegal']

    model_dir = 'trained_models/23092023_CoreEncoder_allregions_LEO_update_loss_augm'
    model_name = 'CoreEncoder'
    model = Core_tiny(input_dim=10, output_dim=31+3+2,)
    # model = torch.compile(model)

    # load model
    best_sd = torch.load(os.path.join(model_dir, f"{model_name}_best.pt"))
    model.load_state_dict(best_sd)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    tiles =   ['/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_07/S2A_MSIL2A_20220727T134711_N0400_R024_T21KXA_20220727T213102_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_04/S2A_MSIL2A_20220414T170851_N0400_R112_T14RPU_20220414T231856_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_04/S2B_MSIL2A_20220413T221939_N0400_R029_T60KYG_20220414T002214_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_10/S2B_MSIL2A_20221020T183409_N0400_R027_T11TPL_20221020T222333_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_04/S2B_MSIL2A_20220429T023239_N0400_R103_T50JLN_20220429T060411_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_01/S2A_MSIL2A_20220122T095321_N0301_R079_T31NGG_20220122T142116_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_01/S2A_MSIL2A_20220129T080201_N0400_R035_T35JLK_20220129T105531_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_01/S2A_MSIL2A_20220110T023101_N0301_R046_T50NMM_20220110T044330_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_10/S2B_MSIL2A_20221002T090709_N0400_R050_T33MTM_20221002T121014_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_07/S2A_MSIL2A_20220706T085611_N0400_R007_T36UVB_20220706T133417_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_01/S2A_MSIL2A_20220128T083221_N0400_R021_T36RUQ_20220128T105623_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_10/S2A_MSIL2A_20221029T181501_N0400_R084_T12RTS_20221029T232858_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_07/S2B_MSIL2A_20220713T093549_N0400_R036_T35VNE_20220713T112940_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_10/S2B_MSIL2A_20221029T090019_N0400_R007_T35TPE_20221029T103554_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_04/S2A_MSIL2A_20220428T164851_N0400_R026_T14QND_20220428T215112_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_07/S2B_MSIL2A_20220729T081609_N0400_R121_T36RXV_20220729T100550_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_07/S2A_MSIL2A_20220729T090601_N0400_R050_T33MTM_20220729T154758_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_07/S2B_MSIL2A_20220704T022539_N0400_R046_T52UDV_20220704T045115_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_10/S2B_MSIL2A_20221030T083009_N0400_R021_T36RUQ_20221030T101025_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_07/S2A_MSIL2A_20220728T013721_N0400_R031_T52KCV_20220728T063155_train_s2.npy', '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_10/S2A_MSIL2A_20221025T134711_N0400_R024_T21KXA_20221025T191904_train_s2.npy']
    #tiles = glob('data_testing/**.tif')
    tiles =[f.split('.npy')[0].split('patches_labeled/')[-1].split('_train_s2')[0] for f in tiles][8:] #['east-africa_10', 'japan_10', 'nigeria_10']

    # tiles = ['10_points_filtered_22_04/S2B_MSIL2A_20220429T023239_N0400_R103_T50JLN_20220429T060411']
    
    # make folder to store results
    results_dir = f'{model_dir}/results'
    os.makedirs(results_dir, exist_ok=True)

    visualise_prediction_tile(tiles,model=model,data_folder='/phileo_data/mini_foundation/mini_foundation_tifs', results_dir=results_dir)

    # x_train, y_train, x_val, y_val, x_test, y_test = data_protocol_bd.protocol_regions(folder=DATA_FOLDER, regions=REGIONS, y='geography')
    
    # _, _, dl_test = load_data(x_train, y_train, x_val, y_val, x_test, y_test,
    #                                     with_augmentations=False,
    #                                     num_workers=0,
    #                                     batch_size=64,
    #                                     encoder_only=True,
    #                                     )
    
    # save_path_visualisations = f"{model_dir}/results/vis.png"
    # metrics = evaluate_model(model, dl_test, device,results_dir, num_visualisations=16*3)


    # with open(f'{results_dir}/{date.today().strftime("%d%m%Y")}_metrics.json', 'w') as fp:
    #     json.dump(metrics, fp)


