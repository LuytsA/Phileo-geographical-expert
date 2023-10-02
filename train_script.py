import torch
import torch.nn as nn
import torchmetrics
import random 
from datetime import date
import numpy as np
import gc 

import sys; sys.path.append("./")
from models.model_CoreCNN_versions import Core_base, Core_tiny, Core_femto
from models.model_Mixer_versions import Mixer_base, Mixer_tiny, Mixer_femto,Mixer_nano

# from models.model_Diamond import DiamondNet
from utils import (
    load_data,
    training_loop,
    data_protocol_bd,
    GeographicalLoss,
    MvMFLoss
)
from utils.training_loop_inf import training_loop_inf
import config_geography
if __name__ == "__main__":

    LOAD_FOLDER = 'trained_models/30092023_Mixer_LEO_geoMvMF_augm'

    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 250
    BATCH_SIZE = 8
    CENTERS = np.load(f"{LOAD_FOLDER}/Mixer_best_centers.npy") #np.load('centers_all_more_enc.npy')
    densities = np.load(f"{LOAD_FOLDER}/Mixer_best_densities.npy")
    NUM_WORKERS = 8
    DATA_FOLDER = '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/' #'/home/andreas/vscode/GeoSpatial/Phileo-geographical-expert/data_geography' #'/home/andreas/vscode/GeoSpatial/Phileo-downstream-tasks/data_landcover' #'/home/lcamilleri/data/s12_buildings/data_patches/'
    print(len(CENTERS))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    torch.cuda.empty_cache()
    # 5 + np.random.rand() for i in range(len(CENTERS))]
    coord_loss = MvMFLoss(center_inits=CENTERS,density_learnable=True, softmax_input=True, density_inits=densities, device=device) 
    kg_loss = nn.CrossEntropyLoss()
    date_loss = nn.MSELoss()
    criterion = GeographicalLoss(coordinate_loss=coord_loss, kg_loss=kg_loss, date_loss=date_loss)
    
    lr_scheduler = None #'reduce_on_plateau' # None, 'reduce_on_plateau', 'cosine_annealing'
    augmentations= True

    #model = Core_tiny(input_dim=10, output_dim=31 + len(CENTERS) + 2)#31+3+2,) #ViT(chw=(10, 64, 64),  n_patches=4, n_blocks=2, hidden_d=768, n_heads=12)# SimpleUnet(input_dim=10, output_dim=1) #ViT(chw=(10, 64, 64),  n_patches=4, n_blocks=2, hidden_d=768, n_heads=12)# SimpleUnet(input_dim=10, output_dim=1) # SimpleUnet()
    model = Mixer_nano(chw = (10,128,128), output_dim=31 + len(CENTERS) + 2, )#31+3+2,) #ViT(chw=(10, 64, 64),  n_patches=4, n_blocks=2, hidden_d=768, n_heads=12)# SimpleUnet(input_dim=10, output_dim=1) #ViT(chw=(10, 64, 64),  n_patches=4, n_blocks=2, hidden_d=768, n_heads=12)# SimpleUnet(input_dim=10, output_dim=1) # SimpleUnet()


    best_sd = torch.load(f'{LOAD_FOLDER}/Mixer_best.pt')
    model.load_state_dict(best_sd)
    model.train()

    del best_sd
    gc.collect()


    NAME = model.__class__.__name__
    OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_LEO_geoMvMF_bugfix'
    OUTPUT_FOLDER = OUTPUT_FOLDER + '_augm' if augmentations else OUTPUT_FOLDER
    if lr_scheduler is not None:
        OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_{lr_scheduler}'
        if lr_scheduler == 'reduce_on_plateau':
            LEARNING_RATE = LEARNING_RATE / 100000 # for warmup start


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
                                          num_workers=NUM_WORKERS,
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
        steps_per_epoch=5000*32//BATCH_SIZE,
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
