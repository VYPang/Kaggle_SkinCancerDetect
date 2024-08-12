import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.model import basicCNN
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import pickle
from data.dataLoading import testSource
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.metrics import ConfusionMatrixDisplay

def testing(model, lossFunction, config, testLoader):
    total_loss = 0
    accurates = 0
    model.eval()
    lossRecord = {i:[] for i in range(10)}
    accRecord = [0 for _ in range(10)]
    countRecord = [0 for _ in range(10)]
    confusionMatrix = np.zeros((10, 10))
    vectorRecord = {}
    test_tqdm = tqdm(testLoader, total=len(testLoader))
    for batch_idx, batch in enumerate(test_tqdm):
        test_tqdm.set_description(f'Testing')

        x, y, mainSet_idx = batch
        y = y.to(device)
        x = x.float().to(device)
        if len(x.shape) == 3:
            x = x[:, None, ...]
        output, vector = model(x)
        vectorRecord[mainSet_idx.cpu().numpy()[0]] = [y.item(), vector.cpu().detach().numpy()[0]]

        # calculate loss
        loss = lossFunction(output, y)
        total_loss += loss

        # calculate accuracy
        _, predicted = torch.max(output, 1)
        accurates += (predicted == y).sum().item()

        # record performance
        lossRecord[y.cpu().numpy()[0]].append(loss.item())
        if predicted == y:
            accRecord[y.item()] += 1
        countRecord[y.item()] += 1
        
        # confusion matrix
        confusionMatrix[y.item(), predicted.item()] += 1
        
    print(f'Over All Loss: {total_loss/len(testLoader)}')
    print(f'Over All Accuracy: {accurates/len(testLoader)}')
    print(f'Worst Group Accuracy: {min(accRecord)/countRecord[np.argmin(accRecord)]}')
    return lossRecord, accRecord, countRecord, vectorRecord, confusionMatrix

def graphPerf(lossRecord, accRecord, countRecord, confusionMatrix, config):
    classes = config.data.classes
    # plot loss histogram
    all_losses = [loss for losses in lossRecord.values() for loss in losses]
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))
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
    disp = ConfusionMatrixDisplay(confusionMatrix.astype(int), display_labels=classes)
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
    classes = config.data.classes
    vectors = np.array([vectorRecord[i][1] for i in range(len(vectorRecord))])
    pca = PCA(n_components=3)
    print('fitting pca...')
    pca.fit(vectors)

    # group vectors by class
    vectorGroup = {i:[] for i in range(10)}
    mainSetIdx = {i:[] for i in range(10)}
    for i in range(len(vectorRecord)):
        vectorGroup[vectorRecord[i][0]].append(vectorRecord[i][1])
        mainSetIdx[vectorRecord[i][0]].append(i)
    
    # plot pca
    fig = go.Figure()
    for i in range(len(vectorGroup)):
        data = pca.transform(np.array(vectorGroup[i]))
        text = [f'mainSetIdx: {idx}' for idx in mainSetIdx[i]]
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
    configPath = 'configuration/config.yaml'
    modelPath = 'ckpt/CIFAR/vote/final.pt'
    config = OmegaConf.load(configPath)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    numClass = len(config.data.classes)
    shape = config.data.shape
    model = basicCNN(shape, numClass, test=True).to(device)
    lossFunction = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(modelPath, map_location=device))

    # load dataset
    testSet = testSource()
    testLoader = DataLoader(testSet, batch_size=1, shuffle=False)

    # test
    lossRecord, accRecord, countRecord, vectorRecord, confusionMatrix = testing(model, lossFunction, config, testLoader)
    graphPerf(lossRecord, accRecord, countRecord, confusionMatrix, config)
    pcaAnalysis(vectorRecord, config)