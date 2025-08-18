import sys
import os

sys.path.append(os.path.abspath(".."))

import numpy as np
import torch
from neuralop.data.datasets.custom_darcy import CustomDarcyDataset, load_darcy_flow
import matplotlib.pyplot as plt
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from copy import deepcopy
import math

#Set resolution, device and losses
resolution = 1024
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


#############################################
## MG-TFNO experiment with 1024 resolution
#############################################


# Create lists of custom Darcy datasets for training and testing
# with multigrid decomposition
num_levels = 3
train_loader_list, test_loader_list, data_processor_list = load_darcy_flow(
    root_dir="data",
    dataset_name="darcy_ZD_PWC",
    n_train=1000,
    n_tests=[50],
    batch_size=100,
    test_batch_sizes=[50],
    train_resolution=resolution,
    test_resolutions=[resolution],
    decompose_multigrid=True,
    L=num_levels
)

print(f"Number of train loaders: {len(train_loader_list)}")

# Train a list of TFNO models, one for each sundomain
# note that in_channels = num_levels + 1 in the model definintion
# TO DO: Parallelize the training of models
models = []
for i in range(len(data_processor_list)):
    print("#####################################################")
    print("#### Train model No.", i)
    print("#####################################################")

    model = TFNO(
        n_modes=(16, 16),
        hidden_channels=64,
        in_channels=num_levels+1,
        out_channels=1,
        factorization='tucker',
        implementation='factorized',
        rank=0.05
    ).to(device)

    n_epochs = 500
    optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    trainer = Trainer(
        model=model,
        n_epochs=n_epochs,
        device=device,
        data_processor=data_processor_list[i],
        eval_interval=20,
        verbose=True
    )

    trainer.train(
        train_loader=train_loader_list[i],
        test_loaders=test_loader_list[i],
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=h1loss,
        eval_losses={'h1': h1loss, 'l2': l2loss}
    )

    models.append(model)

# Directory to save all models
save_dir = "./experiments/tfno_models"
os.makedirs(save_dir, exist_ok=True)

# Save each model
for i, model in enumerate(models):
    model_path = os.path.join(save_dir, f"tfno_model_{i}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model {i} saved at {model_path}")

#############################################
## Normal TFNO experiment with 1024 resolution
#############################################

train_loader, test_loaders, data_processor = load_darcy_flow(
    root_dir="./data/",
    dataset_name='darcy_ZD_PWC',
    n_train=1000,
    n_tests=[50],
    batch_size=1,
    test_batch_sizes=[50],
    train_resolution=resolution,
    test_resolutions=[resolution]
)

tfno_model = TFNO(
    n_modes=(16, 16), 
    hidden_channels=64,
    in_channels=1,
    out_channels=1,
    factorization='tucker',
    implementation='factorized',
    rank=0.05
)
tfno_model = tfno_model.to(device)


n_params = count_model_params(tfno_model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()

n_epochs = 500
optimizer = AdamW(tfno_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)


trainer = Trainer(model=tfno_model, 
                  n_epochs=n_epochs,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=True,
                  eval_interval=50,
                  use_distributed=False,
                  verbose=True)

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)


save_dir = "./experiments/tfno_model_1024"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "tfno_model.pth")
torch.save(tfno_model.state_dict(), model_path)
print(f"Model saved at {model_path}")