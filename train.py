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
from torcheval.metrics.functional import binary_auroc
import os

def train(savePath, device, config, trainLoader, valLoader=None):
    model = ISICModel(config.pretrain.model_name, checkpoint_path=config.pretrain.checkpoint_path).to(device)
    lossFunction = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
    scheduler = fetch_scheduler(optimizer, config)

    epochs = config.train.epochs
    val_auroc_record = 0
    for epoch in range(epochs):
        # training loop
        total_loss = 0
        total_auroc = 0
        dataset_size = 0
        train_tqdm = tqdm(trainLoader, total=len(trainLoader))
        for batch_idx, batch in enumerate(train_tqdm):
            train_tqdm.set_description(f'Epoch {epoch+1}/{epochs}')
            model.train()
            optimizer.zero_grad()

            x = batch['image'].to(device, dtype=torch.float)
            y = batch['target'].to(device, dtype=torch.float)

            output = model(x).squeeze()
            loss = lossFunction(output, y) / config.train.n_accumulate
            auroc = binary_auroc(output, y)
            loss.backward()
            if (batch_idx + 1) % config.train.n_accumulate == 0:
                optimizer.step()
                # zero the parameter gradients
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss * len(y)
            total_auroc += auroc * len(y)
            dataset_size += len(y)
            average_loss = round((total_loss.detach().cpu().numpy()/dataset_size), 5)
            average_auroc = round((total_auroc.detach().cpu().numpy()/dataset_size), 5)
            train_tqdm.set_postfix(train_loss=average_loss, train_auroc=average_auroc)

        # validation loop
        if valLoader != None:
            total_val_loss = 0
            total_val_auroc = 0
            val_dataset_size = 0
            val_tqdm = tqdm(valLoader, total=len(valLoader))
            for batch_idx, batch in enumerate(val_tqdm):
                val_tqdm.set_description('Validation')
                model.eval()
                x = batch['image'].to(device, dtype=torch.float)
                y = batch['target'].to(device, dtype=torch.float)
                with torch.no_grad():
                    output = model(x).squeeze()
                    loss = lossFunction(output, y) / config.train.n_accumulate
                total_val_loss += loss * len(y)
                total_val_auroc += binary_auroc(output, y) * len(y)
                val_dataset_size += len(y)
                average_val_loss = total_val_loss.detach().cpu().numpy()/val_dataset_size
                average_val_auroc = total_val_auroc.detach().cpu().numpy()/val_dataset_size
                val_tqdm.set_postfix(val_loss=round(average_val_loss, 5), val_auroc=round(average_val_auroc, 5))
            if average_val_auroc > val_auroc_record:
                torch.save(model.state_dict(), savePath + f'/val_auroc-{average_val_auroc}.pt')
                print(f'Validation auroc improved from {val_auroc_record} to {average_val_auroc}. Model saved.')
                val_auroc_record = average_val_auroc
        elif (epoch+1) % config.train.save_interval == 0:
            torch.save(model.state_dict(), savePath + f'/epoch{epoch+1}-{average_loss}.pt')
        print('\n')
    torch.save(model.state_dict(), savePath + f'/final.pt')
    print('Final train auc-roc:', average_auroc)
    print('Final train loss:', average_loss)
    if valLoader != None:
        print('Final val auc-roc:', average_val_auroc)
        print('Final val loss:', average_val_loss)

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