import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataLoading import obtain_dataSet
import pandas as pd
from utils.model import ISICModel
from utils.seed import set_seed
from omegaconf import OmegaConf

def infer(model, trainLoader, testLoader):
    trainSet_result, testSet_result = {}, {}
    model.eval()
    train_tqdm = tqdm(trainLoader, total=len(trainLoader))
    for _, batch in enumerate(train_tqdm):
        train_tqdm.set_description(f'writing train csv')

        x = batch['image'].to(device, dtype=torch.float)
        y = batch['target'].to(device, dtype=torch.float)
        isic_id = batch['isic_id']

        with torch.no_grad():
            output = model(x)
        output = output.squeeze().cpu().numpy()
        for idx in range(len(isic_id)):
            trainSet_result[isic_id[idx]] = output[idx]

    test_tqdm = tqdm(testLoader, total=len(testLoader))
    for _, batch in enumerate(test_tqdm):
        test_tqdm.set_description(f'writing test csv')

        x = batch['image'].to(device, dtype=torch.float)
        isic_id = batch['isic_id']

        with torch.no_grad():
            output = model(x)
        output = output.squeeze().cpu().numpy()
        for idx in range(len(isic_id)):
            testSet_result[isic_id[idx]] = output[idx]
    return trainSet_result, testSet_result

if __name__ == "__main__":
    configPath = 'config/default_config.yaml'
    modelPath = 'ckpt/2024-08-25_175314/epoch36-0.20367-0.15668.pt'
    train_df_path = "./data/train-metadata.csv"
    train_hdf_path = "./data/train-image.hdf5"
    test_df_path = "./data/test-metadata.csv"
    test_hdf_path = "./data/test-image.hdf5"

    config = OmegaConf.load(configPath)
    set_seed(config.seed)
    train_df = pd.read_csv(train_df_path)
    config.dataset.val_split = 0 # no validation set
    train_dataset, _ = obtain_dataSet(train_df, config, train_hdf_path)
    trainLoader = DataLoader(train_dataset, batch_size=config.test.batch_size, shuffle=False, pin_memory=True)

    test_df = pd.read_csv(test_df_path)
    test_dataset, _ = obtain_dataSet(test_df, config, test_hdf_path, test=True)
    testLoader = DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=False, pin_memory=True)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = ISICModel(config.pretrain.model_name)
    model.load_state_dict(torch.load(modelPath))
    model.to(device)
    trainSet_result, testSet_result = infer(model, trainLoader, testLoader)
    train_df['cvPred_prob'] = train_df['isic_id'].map(trainSet_result)
    test_df['cvPred_prob'] = test_df['isic_id'].map(testSet_result)
    train_df.to_csv('train-new-metadata.csv', index=False)
    test_df.to_csv('test-new-metadata.csv', index=False)