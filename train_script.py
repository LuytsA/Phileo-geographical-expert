import torch
import torch.nn as nn
import torchmetrics
import random 
from datetime import date
import numpy as np

import sys; sys.path.append("./")
from models.model_CoreCNN import Core_base, Core_tiny, Core_femto
from models.model_Diamond import DiamondNet
from utils import (
    load_data,
    training_loop,
    TiledMSE,
    data_protocol_bd,
    GeographicalLoss,
    MvMFLoss
)
from utils.training_loop_inf import training_loop_inf
import config_geography
CENTERS = np.load('centers_all_more_enc.npy')
print(len(CENTERS))
if __name__ == "__main__":

    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 250
    BATCH_SIZE = 32
    num_workers = 6
    # split_percentage=0.02
    REGIONS = ['denmark-1', 'denmark-2', 'east-africa', 'egypt-1', 'eq-guinea', 'europe', 'ghana-1', 'isreal-1', 'isreal-2', 'japan', 'nigeria','north-america', 'senegal', 'south-america', 'tanzania-1','tanzania-2', 'tanzania-3', 'tanzania-4', 'tanzania-5', 'uganda-1']
    DATA_FOLDER = '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/' #'/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/data_geography' #'/home/andreas/vscode/GeoSpatial/Phileo-downstream-tasks/data_landcover' #'/home/lcamilleri/data/s12_buildings/data_patches/'
    criterion = MvMFLoss(n_center=len(CENTERS), direction_inits=CENTERS, density_inits=[4 + np.random.rand() for i in range(len(CENTERS))]) #GeographicalLoss(n_classes=len(config_geography.regions.keys()))#nn.MSELoss() #nn.CrossEntropyLoss() # vit_mse_losses(n_patches=4)
    lr_scheduler = None #'reduce_on_plateau' # None, 'reduce_on_plateau', 'cosine_annealing'
    augmentations= False

    model = Core_tiny(input_dim=10, output_dim=len(CENTERS))#31+3+2,) #ViT(chw=(10, 64, 64),  n_patches=4, n_blocks=2, hidden_d=768, n_heads=12)# SimpleUnet(input_dim=10, output_dim=1) #ViT(chw=(10, 64, 64),  n_patches=4, n_blocks=2, hidden_d=768, n_heads=12)# SimpleUnet(input_dim=10, output_dim=1) # SimpleUnet()

    # model = DiamondNet(
    #     input_dim=10,
    #     output_dim=3+8,
    #     input_size=64,
    #     depths=[3, 3, 3, 3],
    #     dims=[40, 80, 160, 320],
    #     encoder_only=True)
    # best_sd = torch.load('/home/andreas/vscode/GeoSpatial/Phileo-downstream-tasks/trained_models/10072023_ConvNextV2Unet/ConvNextV2Unet_last.pt')
    # model.load_state_dict(best_sd)


    NAME = model.__class__.__name__
    OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_LEO_MvMF_expdens'
    OUTPUT_FOLDER = OUTPUT_FOLDER + '_augm' if augmentations else OUTPUT_FOLDER
    if lr_scheduler is not None:
        OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_{lr_scheduler}'
        if lr_scheduler == 'reduce_on_plateau':
            LEARNING_RATE = LEARNING_RATE / 100000 # for warmup start

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    # model = torch.compile(model)#, mode='reduce-overhead')
    # x_train, y_train, x_val, y_val, x_test, y_test = data_protocol_bd.protocol_all(folder=DATA_FOLDER,
    #                                                                                     # regions=REGIONS,
    #                                                                                     y='geo',
    #                                                                                     # split_percentage=split_percentage
    #                                                                                     )
    from glob import glob
    import os
    import buteo as beo
    import numpy as np
    x_train_files = sorted(glob(os.path.join(DATA_FOLDER, f"*/*train_s2.npy")))
    y_train_files = sorted(glob(os.path.join(DATA_FOLDER, f"*/*train_label_geo.npy")))

    assert len(x_train_files)==len(y_train_files)

    random.Random(12345).shuffle(x_train_files)
    random.Random(12345).shuffle(y_train_files)
    split_point = int(len(x_train_files)*0.05)
    print(split_point, len(x_train_files))


    x_val_files = x_train_files[:split_point]
    y_val_files = y_train_files[:split_point]
    # x_val_files = x_train_files[split_point:][:2]
    # y_val_files = y_train_files[split_point:][:2]

    x_train_files = x_train_files[split_point:]
    y_train_files = y_train_files[split_point:]

    x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
    y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files])

    x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
    y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files])


    dl_train, dl_val, dl_test = load_data(x_train, y_train, x_val, y_val, x_val, y_val,
                                          with_augmentations=augmentations,
                                          num_workers=num_workers,
                                          batch_size=BATCH_SIZE,
                                          encoder_only=True,
                                          )
    
    # dl_train, dl_val, dl_test = load_data(x_train, y_train, x_val, y_val, x_test, y_test,
    #                                       with_augmentations=augmentations,
    #                                       num_workers=num_workers,
    #                                       batch_size=BATCH_SIZE,
    #                                       encoder_only=True,
    #                                       )

    wmape = torchmetrics.WeightedMeanAbsolutePercentageError(); wmape.__name__ = "wmape"
    mae = torchmetrics.MeanAbsoluteError(); mae.__name__ = "mae"
    mse = torchmetrics.MeanSquaredError(); mse.__name__ = "mse"

    training_loop_inf(
        #num_epochs=NUM_EPOCHS,
        steps_per_epoch=7500,
        learning_rate=LEARNING_RATE,
        model=model,
        criterion=criterion,
        device=device,
        # metrics=[
        #     mse.to(device),
        #     wmape.to(device),
        #     mae.to(device),
        # ],
        metrics=[],
        lr_scheduler=lr_scheduler,
        train_loader=dl_train,
        val_loader=dl_val,
        test_loader=dl_test,
        name=NAME,
        out_folder=OUTPUT_FOLDER,
        predict_func=None,
    )
