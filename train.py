import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataLoading import obtain_dataSet
from utils.model import ISICModel
from utils.seed import set_seed
import torch.optim as optim
import datetime
from omegaconf import OmegaConf
import pandas as pd
from torch.optim import lr_scheduler
import os

def train(savePath, device, config, trainLoader, valLoader=None):
    model = ISICModel(config.pretrain.model_name, checkpoint_path=config.pretrain.checkpoint_path).to(device)
    lossFunction = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
    scheduler = fetch_scheduler(optimizer, config)

    epochs = config.train.epochs
    for epoch in range(epochs):
        # training loop
        total_loss = 0
        train_tqdm = tqdm(trainLoader, total=len(trainLoader))
        for batch_idx, batch in enumerate(train_tqdm):
            train_tqdm.set_description(f'Epoch {epoch+1}/{epochs}')
            model.train()
            optimizer.zero_grad()

            x = batch['image'].to(device, dtype=torch.float)
            y = batch['target'].to(device, dtype=torch.float)

            output = model(x).squeeze()
            loss = lossFunction(output, y) / config.train.n_accumulate
            loss.backward()
            if (batch_idx + 1) % config.train.n_accumulate == 0:
                optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss
            average_loss = round((total_loss.detach().cpu().numpy()/ (batch_idx + 1)), 5)
            train_tqdm.set_postfix(loss=average_loss)

        # validation loop
        if valLoader != None:
            total_val_loss = 0
            val_tqdm = tqdm(valLoader, total=len(valLoader))
            for batch_idx, batch in enumerate(val_tqdm):
                val_tqdm.set_description('Validation')
                model.eval()
                x = batch['image'].to(device, dtype=torch.float)
                y = batch['target'].to(device, dtype=torch.float)
                with torch.no_grad():
                    output = model(x).squeeze()
                    loss = lossFunction(output, y) / config.train.n_accumulate
                total_val_loss += loss
                average_val_loss = round((total_val_loss.detach().cpu().numpy()/ (batch_idx + 1)), 5)
                val_tqdm.set_postfix(val_loss=average_val_loss)
        if epoch % config.train.save_interval == 0:
            torch.save(model.state_dict(), savePath + f'/epoch{epoch+1}-{average_loss}.pt')
    torch.save(model.state_dict(), savePath + f'/final.pt')

def fetch_scheduler(optimizer, config):
    if config.scheduler.name == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.scheduler.T_max, 
                                                   eta_min=config.scheduler.min_lr)
    elif config.scheduler.name == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=config.scheduler.T_0, 
                                                             eta_min=config.scheduler.min_lr)
    elif config.scheduler == None:
        return None
        
    return scheduler

if __name__ == "__main__":
    config_path = "./config/default_config.yaml"
    train_df_path = "./data/train-metadata.csv"
    train_hdf_path = "./data/train-image.hdf5"

    config = OmegaConf.load(config_path)
    set_seed(config.seed)
    train_df = pd.read_csv(train_df_path)
    train_dataset, val_dataset = obtain_dataSet(train_df, config, train_hdf_path)
    trainLoader = DataLoader(train_dataset, batch_size=config.train.train_batch_size, shuffle=False, sampler=train_dataset.sampler)
    if val_dataset != None:
        valLoader = DataLoader(val_dataset, batch_size=config.train.val_batch_size, shuffle=False)
    else:
        valLoader = None

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # create checkpoint folder
    current_time = datetime.datetime.now()
    file_name = current_time.strftime("%Y-%m-%d_%H%M%S")
    savePath = f'ckpt/{file_name}'
    os.makedirs(savePath)
    
    # load model
    train(savePath, device, config, trainLoader, valLoader=valLoader)