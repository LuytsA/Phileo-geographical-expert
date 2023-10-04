import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys; sys.path.append("./")

import matplotlib.pyplot as plt
import buteo as beo
import tqdm


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys; sys.path.append("./")

import matplotlib.pyplot as plt
import buteo as beo
import tqdm

import config_geography
pos_feature_pred = config_geography.feature_positions_predictions
pos_feature_label = config_geography.feature_positions_label




def decode_date(encoded_date):
    doy_sin,doy_cos = encoded_date
    doy = np.arctan2((2*doy_sin-1),(2*doy_cos-1))*365/(2*np.pi)
    if doy<1:
        doy+=365
    return np.array([np.round(doy)])


def decode_coordinates(encoded_coords):
    lat_enc,long_sin,long_cos = encoded_coords
    lat = -lat_enc*180+90
    long = np.arctan2((2*long_sin-1),(2*long_cos-1))*360/(2*np.pi)
    return np.array([lat,long])

def encode_coordinates(coords):
    lat,long = coords
    lat = (-lat + 90)/180
    long_sin = (np.sin(long*2*np.pi/360)+1)/2
    long_cos = (np.cos(long*2*np.pi/360)+1)/2

    return np.array([lat,long_sin,long_cos], dtype=np.float32)



def render_s2_as_rgb(arr, channel_first=False):
    # If there are nodata values, lets cast them to zero.
    if np.ma.isMaskedArray(arr):
        arr = np.ma.getdata(arr.filled(0))

    if channel_first:
        arr = beo.channel_first_to_last(arr)
    # Select only Blue, green, and red. Then invert the order to have R-G-B
    rgb_slice = arr[:, :, 0:3][:, :, ::-1]

    # Clip the data to the quantiles, so the RGB render is not stretched to outliers,
    # Which produces dark images.
    rgb_slice = np.clip(
        rgb_slice,
        np.quantile(rgb_slice, 0.02),
        np.quantile(rgb_slice, 0.98),
    )

    # The current slice is uint16, but we want an uint8 RGB render.
    # We normalise the layer by dividing with the maximum value in the image.
    # Then we multiply it by 255 (the max of uint8) to be in the normal RGB range.
    rgb_slice = (rgb_slice / rgb_slice.max()) * 255.0

    # We then round to the nearest integer and cast it to uint8.
    rgb_slice = np.rint(rgb_slice).astype(np.uint8)

    return rgb_slice


def visualise(x, y, y_pred=None, images=5, channel_first=False, vmin=0, vmax=1, save_path=None, centers=None):
    print(y.shape, y_pred.shape, images)
    rows = images
    if y_pred is None:
        columns = 1
    else:
        columns = 1
    i = 0
    fig = plt.figure(figsize=(10 * columns, 10 * rows))

    for idx in range(0, images):
        arr = x[idx]
        rgb_image = render_s2_as_rgb(arr, channel_first)

        i = i + 1
        fig.add_subplot(rows, columns, i)

        coord_pred_center, coord_true = y_pred[idx,pos_feature_pred['coords']], y[idx,pos_feature_label['coords']]
        kg_pred_logits,kg_true = y_pred[idx,pos_feature_pred['kg']], y[idx,pos_feature_label['kg']]
        date_pred,date_true = y_pred[idx,pos_feature_pred['date']], y[idx,pos_feature_label['date']]
        # coord_true = y[idx,pos_feature_label['coords']]

        # print(y_pred[idx])
        # print(y_pred[idx].shape)



        nearest_center = centers[np.argmax(coord_pred_center)]

        # c_soft = np.exp(coord_pred_center[idx])/np.sum(np.exp(coord_pred_center[idx]),keepdims=True)
        # nearest_center_soft = centers[np.argmax(c_soft)]
        lat_pred,long_pred = decode_coordinates(nearest_center)#coord_pred)
        

        lat,long = decode_coordinates(coord_true)
        doy_pred, doy = decode_date(date_pred), decode_date(date_true)
        climate_pred = config_geography.kg_map[int(np.argmax([kg_pred_logits]))]['climate_class_str']
        climate = config_geography.kg_map[int(np.argmax([kg_true]))]['climate_class_str']
        s1 = f"pred  : lat-long = {np.round(lat_pred,2),np.round(long_pred,2)} \n climate - {climate_pred} \n DoY - {doy_pred}"
        s2 = f"target: lat-long = {np.round(lat,2),np.round(long,2)} \n climate - {climate} \n DoY - {doy}"

        plt.text(25, 25, s1,fontsize=18, bbox=dict(fill=True))
        plt.text(25, 45, s2,fontsize=18, bbox=dict(fill=True))
        plt.imshow(rgb_image)
        plt.axis('on')
        plt.grid()

        # i = i + 1
        # fig.add_subplot(rows, columns, i)
        # plt.imshow(y[idx], vmin=vmin, vmax=vmax, cmap='magma')
        # plt.axis('on')
        # plt.grid()

        # if y_pred is not None:
        #     i = i + 1
        #     fig.add_subplot(rows, columns, i)
        #     plt.imshow(y_pred[idx], vmin=vmin, vmax=vmax, cmap='magma')
        #     plt.axis('on')
        #     plt.grid()

    fig.tight_layout()

    del x
    del y
    del y_pred

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def clip_kg_map(
        kg:str,
        reference: str
    ):
    bbox_ltlng = beo.raster_to_metadata(reference)['bbox_latlng']
    bbox_vector = beo.vector_from_bbox(bbox_ltlng, projection=kg)
    bbox_vector_buffered = beo.vector_buffer(bbox_vector, distance=0.1)
    kg_clipped = beo.raster_clip(kg, bbox_vector_buffered, to_extent=True, adjust_bbox=False)
    kg_aligned = beo.raster_align(kg_clipped,reference=reference,method='reference', resample_alg='nearest')

    beo.delete_dataset_if_in_memory_list([kg_clipped,bbox_vector,bbox_vector_buffered])

    return kg_aligned


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