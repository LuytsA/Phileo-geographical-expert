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
import cv2
import config_geography
import cartopy.crs as crs
import cartopy.feature as cfeature

from utils import (
    load_data,
    data_protocol_bd,
    visualise,
    render_s2_as_rgb,
    decode_coordinates,
)
torch.set_float32_matmul_precision('high')

def precision_recall(y_pred_classification,y_test_classification):
    diff = y_pred_classification.astype(int)-y_test_classification.astype(int)
    fp = np.count_nonzero(diff == 1) # false positives
    fn = np.count_nonzero(diff == -1) # false negatives
    tp = np.sum(y_test_classification) - fn # true positives = all positives - false negatives

    return np.array([tp, fp, fn])


def visualise_prediction_tile(tiles, model, data_folder = '/home/andreas/vscode/GeoSpatial/phi-lab-rd/data/road_segmentation/images', label='lc'):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for p in tiles:
        s2_arr = beo.raster_to_array(f'{data_folder}/{p}_0.tif')[:,:,[1,2,3,4,5,6,7,8,9,10]].astype(np.float32)
        region = p
        coord_bbox = beo.raster_to_metadata(f'{data_folder}/{p}_0.tif')['bbox_latlng']

        patches = beo.array_to_patches(s2_arr,tile_size=64,n_offsets=0)
        np.divide(patches, 10000.0, out=patches)
        patches = beo.channel_last_to_first(patches)

        # x_tile = beo.MultiArray([patches])
        # y_dummy = beo.MultiArray([np.zeros(patches.shape[0])])
        # dl,_,_ = load_data(x_tile, y_dummy,[0],[0],[0],[0],with_augmentations=False,batch_size=16, encoder_only=True)
        dl = DataLoader(patches, batch_size=16, shuffle=False, num_workers=0, drop_last=True, generator=torch.Generator(device='cuda'))


        coords_preds = []
        region_preds = []
        with torch.no_grad():
            for inputs in tqdm(dl):
                batch_pred = model(inputs.to(device))

                y_p = batch_pred.detach().cpu().numpy()

                coord_pred,region_pred = y_p[:,:3], y_p[:,3:].argmax(axis=1)        
                coord_pred = np.array([decode_coordinates(co) for co in coord_pred])

                coords_preds.append(coord_pred)
                region_preds.append(region_pred)
        region_preds, coords_preds = np.array(region_preds),np.concatenate(coords_preds)
        
        rows,columns =2,2
        fig = plt.figure(figsize=(8 * columns, 8 * rows))

        fig.add_subplot(rows, columns, 1)
        rgb_image = render_s2_as_rgb(s2_arr, channel_first=False)
        plt.imshow(rgb_image)
        plt.title(f'{region}')

        fig.add_subplot(rows, columns, 2)
        plt.hist([config_geography.region_inv[int(pred)] for pred in region_preds.flatten()])
        plt.title(f'Predicted region on patches')
        # plt.savefig(f'region_dist_{p}.png')
        # plt.close()


        ax = fig.add_subplot(rows, columns, (3,4), projection=crs.Robinson())
        #ax = fig.add_subplot(1,1,1, projection=crs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.gridlines()
        plt.scatter(x=coords_preds[:,1], y=coords_preds[:,0],
                    color="dodgerblue",
                    s=1,
                    alpha=0.5,
                    transform=crs.PlateCarree()) ## Important

        plt.scatter(x=coord_bbox[2:], y=coord_bbox[:2],
                    color="red",
                    s=16,
                    transform=crs.PlateCarree()) ## Important
        plt.title('coordinate predictions on patches')

        
        fig.tight_layout()
        plt.savefig(f'tile_pred_{p}.png')
        plt.close('all')
        



def grad_cam_vis(model, img, label):
        pred = model(img)#.argmax(dim=1)

        pred_class = pred.argmax(dim=1)
        pred[:,pred_class].backward()
        gradients = model.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = model.get_activations(img).detach()

        for i in range(768):
            activations[:, i, :, :] *= pooled_gradients[i]
        print(gradients.shape, pooled_gradients.shape, activations.shape)

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        import matplotlib.pyplot as plt
        plt.matshow(heatmap.squeeze())
        # plt.savefig('gradcam.png')

        # print(img.shape,label.shape)
        # visualise([img.numpy()[0]], [label.numpy()[0]], y_pred=pred.argmax(dim=1),channel_first=True, images=1, class_dict=None, save_path='vis_grad.png')

        img = render_s2_as_rgb(img.numpy()[0], channel_first=True)
        # img = cv2.imread('./data/Elephant/data/05fig34.jpg')
        heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

        fig = plt.figure(figsize=(10 * 1, 10 * 1))
        fig.add_subplot(1, 3, 1)
        plt.imshow(img)
        fig.add_subplot(1, 3, 2)
        plt.imshow(heatmap)
        fig.add_subplot(1, 3, 3)
        plt.imshow(superimposed_img/255)
        plt.savefig(f'heatmap_{j}.png')


def evaluate_model(model, dataloader_test, device, result_dir, num_visualisations = 20):
    num_classes=len(config_geography.regions.keys())
    torch.set_default_device(device)
    model.to(device)
    model.eval()
    
    confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to('cpu')
    running_conf = np.zeros((num_classes,num_classes))
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

            coord_pred,region_pred = y_p[:,:3], y_p[:,3:].argmax(axis=1)
            coord_true, region = y_t[:,:3],y_t[:,3:].squeeze()

            #print(region.shape,region_pred.shape)
            
            #print([(decode_coordinates(cp)-decode_coordinates(ct))**2 for cp,ct in zip(coord_pred,coord_true)])
            ss = np.array([(decode_coordinates(cp)-decode_coordinates(ct))**2 for cp,ct in zip(coord_pred,coord_true)])
            s = np.sum(ss,axis=0)
            # print(ss.shape, s.shape)
            mse_coords += s
            running_conf += confmat(region_pred,region).numpy()

            if len(x_true)<num_visualisations:
                x_true.append(inputs[:1, 0:3, :, :].detach().cpu().numpy()) # only a few per batch to avoid memory issues
                y_pred.append(y_p.numpy())
                y_true.append(y_t.numpy())
    
    y_pred = np.concatenate(y_pred,axis=0)
    x_true = np.concatenate(x_true,axis=0)
    y_true = np.concatenate(y_true,axis=0)

    total = np.sum(running_conf)
    tp = running_conf.trace()

    s = np.sum(running_conf, axis=1, keepdims=True)
    s[s==0]=1
    running_conf = running_conf/s
    
    save_path_visualisations = f"{result_dir}/vis.png"
    plt.figure(figsize = (12,9))
    ax = sn.heatmap(running_conf, annot=True, fmt='.2f')
    ax.xaxis.set_ticklabels(config_geography.region_inv.values(),rotation = 90)
    ax.yaxis.set_ticklabels(config_geography.region_inv.values(),rotation = 0)
    plt.savefig(save_path_visualisations.replace('vis','cm'))
    
    
    batch_size = dataloader_test.batch_size
    #y_vis = [i*batch_size for i in range(0,num_visualisations)]
    metrics = {'acc_regions':tp/total, 'rmse_lat':np.sqrt(mse_coords[0]/(batch_size*len(dataloader_test))),'rmse_long':np.sqrt(mse_coords[1]/(batch_size*len(dataloader_test))),'total_regions':total}

    #plt.savefig(save_path_visualisations.replace("vis","cm22"))
    visualise(x_true, np.squeeze(y_true), np.squeeze(y_pred), images=num_visualisations, channel_first=True, vmin=0, vmax=0.5, save_path=save_path_visualisations)
    print(metrics)
    #return metrics
        


if __name__ == "__main__":

    DATA_FOLDER = '/home/andreas/vscode/GeoSpatial/Phileo-downstream-tasks/data_landcover'
    REGIONS = ['north-america','east-africa', 'europe','eq-guinea', 'japan','south-america', 'nigeria', 'senegal']

    model_dir = 'trained_models/30082023_CoreEncoder_allregions_split1'
    model_name = 'CoreEncoder'
    model = Core_tiny(input_dim=10, output_dim=3+8,)
    model = torch.compile(model)

    # load model
    best_sd = torch.load(os.path.join(model_dir, f"{model_name}_best.pt"))
    model.load_state_dict(best_sd)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)


    # tiles = glob('data/**.tif')
    # tiles =['isreal-1_1'] #[f.split('.tif')[0].split('/')[-1].split('_0')[0] for f in tiles] #['east-africa_10', 'japan_10', 'nigeria_10']
    # visualise_prediction_tile(tiles,model=model,data_folder='data/')
    
    # make folder to store results
    results_dir = f'{model_dir}/results'
    os.makedirs(results_dir, exist_ok=True)
    results = {}


    x_train, y_train, x_val, y_val, x_test, y_test = data_protocol_bd.protocol_regions(folder=DATA_FOLDER, regions=REGIONS, y='geography')
    
    _, _, dl_test = load_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                        with_augmentations=False,
                                        num_workers=0,
                                        batch_size=64,
                                        encoder_only=True,
                                        )
    
    save_path_visualisations = f"{model_dir}/results/vis.png"
    metrics = evaluate_model(model, dl_test, device,results_dir, num_visualisations=16*3)


    with open(f'{results_dir}/{date.today().strftime("%d%m%Y")}_metrics.json', 'w') as fp:
        json.dump(results, fp)

