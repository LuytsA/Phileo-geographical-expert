# general 
import random 
from datetime import date
import numpy as np
from glob import glob
import os
import buteo as beo
import numpy as np
import sys; sys.path.append("./")

#torch
import torch
import torch.nn as nn
import torchmetrics

#models 
from models.model_CoreCNN_versions import Core_base, Core_tiny, Core_femto, Core_nano
from models.model_Mixer_versions import Mixer_base, Mixer_tiny, Mixer_femto, Mixer_nano

# utils 
from utils import load_data, GeographicalLoss
from utils.training_loop_inf import training_loop_inf
from utils.MvMF_utils import MvMFLoss
import config_geography
from utils.encoding_utils import decode_coordinates

if __name__ == "__main__":
    
    #cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    torch.cuda.empty_cache()
    
    # general
    DATA_FOLDER = '/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/'
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 250
    BATCH_SIZE = 32
    NUM_WORKERS = 6
    LR_SCHEDULER = None # None, 'reduce_on_plateau', 'cosine_annealing'
    AUGMENTATIONS= True

    # geographic loss
    if config_geography.USE_MvMFloss:
        CENTERS = config_geography.centers
        DENSITIES = [7 + 2*np.random.rand() for i in range(len(CENTERS))]
        DENSITY_LEARNABLE = True
        CENTERS_LEARNABLE = False
        
        coord_loss = MvMFLoss(center_inits=CENTERS, density_inits=DENSITIES, density_learnable=DENSITY_LEARNABLE, centers_learnable=CENTERS_LEARNABLE, softmax_input=True, device=device) 
        output_dim = 31+len(CENTERS)+2
    else:
        coord_loss = nn.MSELoss()
        output_dim = 31+3+2

    kg_loss = nn.CrossEntropyLoss()
    date_loss = nn.MSELoss()
    criterion = GeographicalLoss(coordinate_loss=coord_loss, kg_loss=kg_loss, date_loss=date_loss)
    output_dim = 31+3+2 if type(coord_loss)==nn.MSELoss else  31+len(CENTERS)+2

    # model
    model = Core_nano(input_dim=10, output_dim=output_dim)
    # model = Mixer_nano(chw = (10,128,128), output_dim=31 + len(CENTERS) + 2)
    model.train()


    NAME = model.__class__.__name__
    OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_LEO_geoMvMF'
    OUTPUT_FOLDER = OUTPUT_FOLDER + '_augm' if AUGMENTATIONS else OUTPUT_FOLDER
    if LR_SCHEDULER is not None:
        OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_{LR_SCHEDULER}'
        if LR_SCHEDULER == 'reduce_on_plateau':
            LEARNING_RATE = LEARNING_RATE / 100000 # for warmup start

    # get train and val files
    x_train_files = sorted(glob(os.path.join(DATA_FOLDER, f"*/*train_s2.npy")))
    y_train_files = sorted(glob(os.path.join(DATA_FOLDER, f"*/*train_label_geo.npy")))
    assert len(x_train_files)==len(y_train_files)

    # random split
    random.Random(12345).shuffle(x_train_files)
    random.Random(12345).shuffle(y_train_files)
    split_point = int(len(x_train_files)*0.05)
    
    x_val_files = x_train_files[:split_point]
    y_val_files = y_train_files[:split_point]
    x_train_files = x_train_files[split_point:]
    y_train_files = y_train_files[split_point:]

    x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
    y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files])
    x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
    y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files])

    # get dataloaders
    dl_train, dl_val, _ = load_data(x_train, y_train, x_val, y_val, x_val, y_val,
                                        with_augmentations=AUGMENTATIONS,
                                        num_workers=NUM_WORKERS,
                                        batch_size=BATCH_SIZE,
                                        encoder_only=True,
                                        )

    training_loop_inf(
        steps_per_epoch=5000*32//BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        model=model,
        criterion=criterion,
        device=device,
        metrics=[],
        lr_scheduler=LR_SCHEDULER,
        train_loader=dl_train,
        val_loader=dl_val,
        name=NAME,
        out_folder=OUTPUT_FOLDER,
    )
