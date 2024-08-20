import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataLoading import obtain_dataSet
import pandas as pd
from utils.model import ISICModel
from utils.seed import set_seed
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.metrics import ConfusionMatrixDisplay

def testing(model, config, testLoader):
    lossFunction = nn.BCELoss()
    total_loss = 0
    accurates = 0
    model.eval()
    lossRecord = {0: [], 1: []}
    accRecord = [0, 0]
    countRecord = [0, 0]
    confusionMatrix = np.zeros((2, 2))
    vectorRecord = {}
    test_tqdm = tqdm(testLoader, total=len(testLoader))
    for batch_idx, batch in enumerate(test_tqdm):
        test_tqdm.set_description(f'Testing')

        x = batch['image'].to(device, dtype=torch.float)
        y = batch['target'].to(device, dtype=torch.float)
        isic_id = batch['isic_id']

        with torch.no_grad():
            output, vector = model(x)

        # calculate loss
        for idx in range(len(isic_id)):
            loss = lossFunction(output[idx].squeeze(), y[idx].squeeze())
            lossRecord[int(y[idx].item())].append(loss.item())
            total_loss += loss.item()

        output = output.squeeze().cpu().numpy()
        y = y.cpu().numpy().astype(int)

        # calculate accuracy
        predicted = (output > 0.5).astype(int)
        accurates += (predicted == y).sum()
        # record belonging class
        count = np.unique(y, return_counts=True)[1]
        for i in range(len(count)):
            countRecord[i] += count[i]

        for idx in range(len(isic_id)):
            vectorRecord[isic_id[idx]] = [y[idx].item(), vector.cpu().detach().numpy()[idx]]
            if predicted[idx] == y[idx]:
                accRecord[y[idx]] += 1
            
            # confusion matrix
            confusionMatrix[y[idx], predicted[idx]] += 1
    
    total_count = np.array(countRecord).sum()
    print(f'Over All Loss: {total_loss/total_count}')
    print(f'Over All Accuracy: {accurates/total_count}')
    print(f'Worst Group Accuracy: {min(accRecord)/countRecord[np.argmin(accRecord)]}')
    return lossRecord, accRecord, countRecord, vectorRecord, confusionMatrix

def graphPerf(lossRecord, accRecord, countRecord, confusionMatrix, config):
    classes = config.dataset.classes
    # plot loss histogram
    all_losses = [loss for losses in lossRecord.values() for loss in losses]
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
    for i, (ax, losses) in enumerate(zip(axes.flat, lossRecord.values())):
        ax.hist(losses, bins=50, range=(min_loss, max_loss))
        ax.set_title(f'Class {classes[i]}')
        ax.set_xlabel('Loss Value')
        ax.set_ylabel('Frequency')
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.savefig('lossHistogram.jpg')
    plt.clf()

    # plot accuracy histogram
    accuracy = [accRecord[i]/countRecord[i] for i in range(len(accRecord))]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(accuracy)), accuracy)
    ax.set_xticks(range(len(accuracy)))
    ax.set_xticklabels([f'{classes[i]}' for i in range(len(accuracy))])
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per Class')
    plt.tight_layout()
    plt.savefig('accuracyBar.jpg')
    plt.clf()

    # graph confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusionMatrix.astype(int), display_labels=classes.values())
    disp.plot()
    disp.plot(ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusionMatrix.jpg', dpi=300)

    # info printing
    for i in range(len(accRecord)):
        print(f'{classes[i]} \t(total counts: {countRecord[i]}\taccuracy: {round(accuracy[i], 4)})')

def pcaAnalysis(vectorRecord, config):
    classes = config.dataset.classes
    vectors = np.array([i[1] for i in vectorRecord.values()])
    pca = PCA(n_components=3)
    print('fitting pca...')
    pca.fit(vectors)

    # group vectors by class
    vectorGroup = {0:[], 1:[]}
    dataIdx = {0:[], 1:[]}
    for isic_id, record in vectorRecord.items():
        vectorGroup[record[0]].append(record[1])
        dataIdx[record[0]].append(isic_id)
    
    # plot pca
    fig = go.Figure()
    for i in range(len(vectorGroup)):
        data = pca.transform(np.array(vectorGroup[i]))
        text = [f'dataId: {idx}' for idx in dataIdx[i]]
        fig.add_trace(go.Scatter3d(text=text,
                                    x=data[:, 0],
                                    y=data[:, 1],
                                    z=data[:, 2],
                                    mode='markers',
                                    marker=dict(size=4),
                                    name=f'class {classes[i]}'
                                   ))
    fig.update_layout(scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ))
    fig.write_html('pca.html')
    print('pca analysis done!')

if __name__ == "__main__":
    configPath = 'config/default_config.yaml'
    modelPath = 'ckpt/2024-08-19_161649/epoch31-0.08709-0.06277.pt'
    test_df_path = "./data/train-metadata.csv"
    test_hdf_path = "./data/train-image.hdf5"

    config = OmegaConf.load(configPath)
    set_seed(config.seed)
    test_df = pd.read_csv(test_df_path)
    test_dataset, _ = obtain_dataSet(test_df, config, test_hdf_path, test=True)
    testLoader = DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=False)
    # testLoader = DataLoader(test_dataset, batch_size=config.test.batch_size, shuffle=False, sampler=test_dataset.sampler) # debug
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = ISICModel(config.pretrain.model_name, test=True)
    model.load_state_dict(torch.load(modelPath))
    model.to(device)

    # test
    lossRecord, accRecord, countRecord, vectorRecord, confusionMatrix = testing(model, config, testLoader)
    graphPerf(lossRecord, accRecord, countRecord, confusionMatrix, config)
    pcaAnalysis(vectorRecord, config)