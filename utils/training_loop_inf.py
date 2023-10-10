# Standard Library
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib
# import PyQt5
# matplotlib.use('QtAgg')
import json

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
# torch.autograd.set_detect_anomaly(True, check_nan=True)

# utils
from utils.visualisations import visualise


def training_loop_inf(
    steps_per_epoch: int,
    learning_rate: float,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    metrics: list = None,
    lr_scheduler: str = None,
    name="model",
    out_folder="trained_models/",
    predict_func=None,
    visualise_validation=True
) -> None:
        
    torch.set_default_device(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("No CUDA device available.")

    print("Starting training...")
    print("")

    model.to(device)
    os.makedirs(out_folder, exist_ok=True)

    # Loss and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-06)
    optimizer_loss = torch.optim.AdamW(criterion.coord_loss.parameters(), lr=1e-04, eps=1e-06)
    scaler = GradScaler()

    # Save the initial learning rate in optimizer's param_groups
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = learning_rate

    if lr_scheduler == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            20,
            2,
            eta_min=0.000001,
        )
    elif lr_scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 4, 5], gamma=(10))
        warmup = True

    best_epoch = 0
    best_loss = None
    best_model_state = model.state_dict().copy()
    epochs_no_improve = 0

    # used for plots
    tl = []
    vl = []
    e = []
    lr = []

    cl=[]
    kgl=[]
    dl=[]

    # Training loop
    epoch = 0 
    train_pbar = tqdm(train_loader, desc=f"Infinite loader with {steps_per_epoch} steps_per_epoch, epoch={epoch}")
    train_loss = 0.0
    train_cl = 0.0
    train_kgl = 0.0
    train_dl = 0.0
    train_metrics_values = { metric.__name__: 0.0 for metric in metrics }
    #with torch.autograd.detect_anomaly():
    for i, (images, labels) in enumerate(train_pbar):
        if torch.isnan(images).sum() > 0:
            np.save(f'debugging/images_nan_{i}.npy', images.detach().cpu().numpy())
        if torch.isinf(images).sum() > 0:
            np.save(f'debugging/images_inf_{i}.npy', images.detach().cpu().numpy())

        if epoch == 5 and lr_scheduler == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, min_lr=1e-6)
            warmup = False
            print('Warmup finished')

        model.train()


        # Move inputs and targets to the device (GPU)
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()
        optimizer_loss.zero_grad()

        # if i%10==0:
        #     print(torch.mean(criterion.densities))
        #     print(torch.mean(criterion.directions))

        # Cast to bfloat16
        with autocast(dtype=torch.bfloat16):
            outputs = model(images)
            loss,coord_loss,kg_loss,date_loss, only_sea = criterion(outputs, labels)
            if only_sea:
                loss = kg_loss
                print('only sea')

            if torch.isnan(loss):
                torch.save(model.state_dict().copy(), f"debugging/{name}_last_{i}.pt")
                np.save(f'debugging/images_{i}.npy', images.detach().cpu().numpy())
                np.save(f'debugging/outputs_{i}.npy', outputs.detach().cpu().numpy())
                np.save(f'debugging/loss_{i}.npy', loss.detach().cpu().numpy())
                np.save(f'debugging/kg_loss_{i}.npy', kg_loss.detach().cpu().numpy())
                np.save(f'debugging/label_{i}.npy', labels.detach().cpu().numpy())


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(optimizer_loss)
            scaler.update()
            # optimizer_loss.step()


        train_loss += loss.item()
        train_cl += coord_loss.item()
        train_kgl += kg_loss.item()
        train_dl += date_loss.item()

        for metric in metrics:
            train_metrics_values[metric.__name__] += metric(outputs, labels)

        train_pbar.set_postfix({
            "loss": f"{train_loss / (i%steps_per_epoch + 1):.4f}","cl": f"{train_cl / (i%steps_per_epoch + 1):.4f}","kgl": f"{train_kgl / (i%steps_per_epoch + 1):.4f}","dl": f"{train_dl / (i%steps_per_epoch + 1):.4f}",
            **{name: f"{value / (i%steps_per_epoch + 1):.4f}" for name, value in train_metrics_values.items()}
        })

        # Validate at the end of each epoch
        # This is done in the same scope to keep tqdm happy.
        if (i+1)%steps_per_epoch == 0:
            
            # start validation
            val_metrics_values = { metric.__name__: 0.0 for metric in metrics }
            with torch.no_grad():
                model.eval()

                val_loss = 0

                stop_j = int(0.01*len(val_loader))

                # # visualise some validation results
                num_visualisations = 20
                if num_visualisations > len(val_loader):
                    num_visualisations = len(val_loader)
                vis_batches = [i for i in range(num_visualisations)]
                # vis_batches = [i*len(val_loader)//(num_visualisations+1) for i in range(num_visualisations)]
                vis_images = []
                vis_labels = []
                vis_preds = []
                save_dir = f'{out_folder}/visualisations/'
                os.makedirs(save_dir, exist_ok=True)

                running_coords=0
                running_kg = 0
                running_date = 0
                n_sea = 0
                for j, (images, labels) in enumerate(val_loader):

                    if j> stop_j:
                        break
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    loss, coord_loss, kg_loss, date_loss, only_sea = criterion(outputs, labels)
                    if only_sea:
                        loss = kg_loss
                        date_loss = torch.tensor([0])
                        coord_loss = torch.tensor([0])
                        n_sea += 1

                    val_loss += loss.item()
                    running_coords += coord_loss.item()
                    running_kg += kg_loss.item()
                    running_date += date_loss.item()



                    # if labels.shape != outputs.shape:
                    #     outputs = unpatchify(labels.shape[0], labels.shape[1], labels.shape[2], labels.shape[3],
                    #                          n_patches=4, tensors=outputs)

                    if j in vis_batches:
                        vis_images.append(images.detach().cpu().numpy()[0])
                        vis_labels.append(labels.detach().cpu().numpy()[0])
                        vis_preds.append(outputs.detach().cpu().numpy()[0])

                    
                    for metric in metrics:
                        val_metrics_values[metric.__name__] += metric(outputs, labels)
            
            print(f'only sea {n_sea} times out of {len(val_loader)}')
            # Append val_loss to the train_pbar
            train_pbar.set_postfix({
                "loss": f"{train_loss / (steps_per_epoch):.4f}",
                **{name: f"{value / (steps_per_epoch):.4f}" for name, value in train_metrics_values.items()},
                "val_l": f"{val_loss / (j + 1):.4f}","c_l": f"{running_coords / (j + 1):.4f}","t_l": f"{running_date / (j + 1):.4f}","kg_l": f"{running_kg / (j + 1):.4f}","t_l": f"{running_date / (j + 1):.4f}",
                **{f"val_{name}": f"{value / (j + 1):.4f}" for name, value in val_metrics_values.items()},
                #f"lr": optimizer.param_groups[0]['lr'],
            }, refresh=True)
            print('')

            tl.append(train_loss / (steps_per_epoch))
            vl.append(val_loss/ (j + 1))
            lr.append(optimizer.param_groups[0]['lr'])

            cl.append(running_coords/ (j + 1))
            kgl.append(running_kg/ (j + 1))
            dl.append(running_date/ (j+ 1))

            # # Update the scheduler
            if lr_scheduler == 'cosine_annealing':
                scheduler.step()
            elif lr_scheduler == 'reduce_on_plateau':
                if warmup:
                    scheduler.step()
                else:
                    scheduler.step(vl[-1])

            # if best_loss is None:
            #     best_epoch = epoch
            #     best_loss = val_loss
            #     best_model_state = model.state_dict().copy()
            #     torch.save(best_model_state, os.path.join(out_folder, f"{name}_best.pt"))
            #     np.save(os.path.join(out_folder, f"{name}_best_centers.npy"), criterion.coord_loss.centers.detach().cpu().numpy())
            #     np.save(os.path.join(out_folder, f"{name}_best_densities.npy"), criterion.coord_loss.densities.detach().cpu().numpy())

            #     if predict_func is not None:
            #         predict_func(model, epoch + 1)

            # elif best_loss > val_loss:
            #     best_epoch = epoch
            #     best_loss = val_loss
            #     best_model_state = model.state_dict().copy()
            #     torch.save(best_model_state, os.path.join(out_folder, f"{name}_best.pt"))
            #     np.save(os.path.join(out_folder, f"{name}_best_centers.npy"), criterion.coord_loss.centers.detach().cpu().numpy())
            #     np.save(os.path.join(out_folder, f"{name}_best_densities.npy"), criterion.coord_loss.densities.detach().cpu().numpy())

            #     if predict_func is not None:
            #         predict_func(model, epoch + 1)

            #     epochs_no_improve = 0
            # else:
            #     epochs_no_improve += 1
            torch.save(model.state_dict().copy(), os.path.join(out_folder, f"{name}_last_{epoch}.pt"))
            np.save(os.path.join(out_folder, f"{name}_last_centers_{epoch}.npy"), criterion.coord_loss.centers.detach().cpu().numpy())
            np.save(os.path.join(out_folder, f"{name}_last_densities_{epoch}.npy"), criterion.coord_loss.densities.detach().cpu().numpy())

            # visualize loss & lr curves
            e.append(epoch)

            fig = plt.figure()
            plt.plot(e, tl, label='Training Loss', )
            plt.plot(e, vl, label='Validation Loss')
            plt.plot(e, cl, label='Coord Loss')
            plt.plot(e, kgl, label='kg Loss')
            plt.plot(e, dl, label='date Loss')

            loss_dict = {'training':tl, 'validation':vl, 'coordinate':cl, 'kg':kgl, 'date':dl}
            with open(os.path.join(out_folder, "losses.json"), 'w') as f:
                json.dump(loss_dict, f)

            # plt.yscale('log')

            plt.legend()
            plt.savefig(os.path.join(out_folder, f"loss.png"))
            plt.close('all')
            fig = plt.figure()
            plt.plot(e, lr, label='Learning Rate')
            plt.legend()
            plt.savefig(os.path.join(out_folder, f"lr.png"))
            plt.close('all')

            if visualise_validation:
                visualise(vis_images, np.squeeze(vis_labels), np.squeeze(vis_preds), images=num_visualisations,
                            channel_first=True, vmin=0, vmax=1, save_path=os.path.join(save_dir, f"val_pred_{epoch}.png"), centers=criterion.coord_loss.centers.detach().cpu().numpy())
            
            # reset train values and up the epoch
            epoch += 1
            # Initialize the running loss
            train_loss = 0.0
            train_loss = 0.0
            train_cl = 0.0
            train_kgl = 0.0
            train_dl = 0.0
            train_metrics_values = { metric.__name__: 0.0 for metric in metrics }
            # Initialize the progress bar for training
            train_pbar.set_description(desc=f"Infinite loader with {steps_per_epoch} steps_per_epoch, epoch={epoch}", refresh=False)


            #
            # # Early stopping
            # if epochs_no_improve == patience_calculator(epoch, t_0, t_mult, max_patience):
            #     print(f'Early stopping triggered after {epoch + 1} epochs.')
            #     break