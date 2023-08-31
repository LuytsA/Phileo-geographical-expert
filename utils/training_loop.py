# Standard Library
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib
# import PyQt5
# matplotlib.use('QtAgg')

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np


# utils
from utils import visualise


def training_loop(
    num_epochs: int,
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
            last_epoch=num_epochs - 1,
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

    # Training loop
    for epoch in range(num_epochs):

        if epoch == 5 and lr_scheduler == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, min_lr=1e-6)
            warmup = False
            print('Warmup finished')

        model.train()

        # Initialize the running loss
        train_loss = 0.0
        train_metrics_values = { metric.__name__: 0.0 for metric in metrics }

        # Initialize the progress bar for training
        train_pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Cast to bfloat16
            with autocast(dtype=torch.float16):
                outputs = model(images)
                loss,_,_ = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            train_loss += loss.item()

            for metric in metrics:
                train_metrics_values[metric.__name__] += metric(outputs, labels)

            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
                **{name: f"{value / (i + 1):.4f}" for name, value in train_metrics_values.items()}
            })

            # Validate at the end of each epoch
            # This is done in the same scope to keep tqdm happy.
            if i == len(train_loader) - 1:

                val_metrics_values = { metric.__name__: 0.0 for metric in metrics }
                # Validate every epoch
                with torch.no_grad():
                    model.eval()

                    val_loss = 0

                    # # visualise some validation results
                    num_visualisations = 20
                    if num_visualisations > len(val_loader):
                        num_visualisations = len(val_loader)
                    vis_batches = [i*len(val_loader)//(num_visualisations+1) for i in range(num_visualisations)]
                    vis_images = []
                    vis_labels = []
                    vis_preds = []
                    save_dir = f'{out_folder}/visualisations/'
                    os.makedirs(save_dir, exist_ok=True)

                    running_region = 0
                    running_coords=0
                    for j, (images, labels) in enumerate(val_loader):
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)

                        loss, region_loss, coord_loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        running_coords += coord_loss.item()
                        running_region += region_loss.item()


                        # if labels.shape != outputs.shape:
                        #     outputs = unpatchify(labels.shape[0], labels.shape[1], labels.shape[2], labels.shape[3],
                        #                          n_patches=4, tensors=outputs)

                        if j in vis_batches:
                            vis_images.append(images.detach().cpu().numpy()[0])
                            vis_labels.append(labels.detach().cpu().numpy()[0])
                            vis_preds.append(outputs.detach().cpu().numpy()[0])
                        
                        for metric in metrics:
                            val_metrics_values[metric.__name__] += metric(outputs, labels)

                # Append val_loss to the train_pbar
                train_pbar.set_postfix({
                    "loss": f"{train_loss / (i + 1):.4f}",
                    **{name: f"{value / (i + 1):.4f}" for name, value in train_metrics_values.items()},
                    "val_loss": f"{val_loss / (j + 1):.4f}","region_loss": f"{running_region / (j + 1):.4f}","coord_loss": f"{running_coords / (j + 1):.4f}",
                    **{f"val_{name}": f"{value / (j + 1):.4f}" for name, value in val_metrics_values.items()},
                    f"lr": optimizer.param_groups[0]['lr'],
                }, refresh=True)
                tl.append(train_loss / (i + 1))
                vl.append(val_loss/ (j + 1))
                lr.append(optimizer.param_groups[0]['lr'])

                # # Update the scheduler
                if lr_scheduler == 'cosine_annealing':
                    scheduler.step()
                elif lr_scheduler == 'reduce_on_plateau':
                    if warmup:
                        scheduler.step()
                    else:
                        scheduler.step(vl[-1])

                if best_loss is None:
                    best_epoch = epoch
                    best_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    torch.save(best_model_state, os.path.join(out_folder, f"{name}_best.pt"))
                    
                    if predict_func is not None:
                        predict_func(model, epoch + 1)

                elif best_loss > val_loss:
                    best_epoch = epoch
                    best_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    torch.save(best_model_state, os.path.join(out_folder, f"{name}_best.pt"))

                    if predict_func is not None:
                        predict_func(model, epoch + 1)

                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                torch.save(best_model_state, os.path.join(out_folder, f"{name}_last.pt"))

        # visualize loss & lr curves
        e.append(epoch)

        fig = plt.figure()
        plt.plot(e, tl, label='Training Loss', )
        plt.plot(e, vl, label='Validation Loss')
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
                      channel_first=True, vmin=0, vmax=1, save_path=os.path.join(save_dir, f"val_pred_{epoch}.png"))



        #
        # # Early stopping
        # if epochs_no_improve == patience_calculator(epoch, t_0, t_mult, max_patience):
        #     print(f'Early stopping triggered after {epoch + 1} epochs.')
        #     break

    # Load the best weights
    model.load_state_dict(best_model_state)

    print("Finished Training. Best epoch: ", best_epoch + 1)
    print("")
    print("Starting Testing...")
    model.eval()

    # Test the model
    with torch.no_grad():
        test_loss = 0
        for k, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

        print(f"Test Accuracy: {test_loss / (k + 1):.4f}")
        if labels.shape != outputs.shape:
            outputs = unpatchify(labels.shape[0], labels.shape[1], labels.shape[2], labels.shape[3],
                                 n_patches=4, tensors=outputs)

        visualise(images.detach().cpu().numpy(), np.squeeze(labels.detach().cpu().numpy()), np.squeeze(outputs.detach().cpu().numpy()), images=num_visualisations,
                  channel_first=True, vmin=0, vmax=1, save_path=os.path.join(save_dir, f"test_pred.png"))

    # Save the model
    torch.save(best_model_state, os.path.join(out_folder, f"{name}.pt"))