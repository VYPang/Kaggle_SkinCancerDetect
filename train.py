import torch
import torch.nn as nn
from tqdm import tqdm
from data.dataLoading import trainSource
from torch.utils.data import DataLoader
from utils.model import basicCNN
import torch.optim as optim
import datetime
from omegaconf import OmegaConf
import numpy as np
import os

def train(savePath, device, lossFunction, config, trainLoader, valLoader=None, saveModel=True):
    numClass = len(config.data.classes)
    shape = config.data.shape
    model = basicCNN(shape, numClass).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.train.lr, momentum=config.train.momentum)

    epochs = config.train.epochs
    for epoch in range(epochs):
        # training loop
        total_loss = 0
        train_tqdm = tqdm(trainLoader, total=len(trainLoader))
        for batch_idx, batch in enumerate(train_tqdm):
            train_tqdm.set_description(f'Epoch {epoch+1}/{epochs}')
            model.train()
            optimizer.zero_grad()

            x, y, _ = batch
            y = y.to(device)
            x = x.float().to(device)
            if len(x.shape) == 3:
                x = x[:, None, ...]

            output = model(x)
            loss = lossFunction(output, y)
            total_loss += loss
            loss.backward()
            optimizer.step()
            average_loss = round((total_loss.detach().cpu().numpy()/ (batch_idx + 1)), 5)
            train_tqdm.set_postfix(loss=average_loss)

        # validation loop
        if valLoader != None:
            total_val_loss = 0
            val_tqdm = tqdm(valLoader, total=len(valLoader))
            for batch_idx, batch in enumerate(val_tqdm):
                val_tqdm.set_description('Validation')
                model.eval()
                x, y, _ = batch
                y = y.to(device)
                x = x.float().to(device)
                output = model(x)
                loss = lossFunction(output, y)
                total_val_loss += loss
                average_val_loss = round((total_val_loss.detach().cpu().numpy()/ (batch_idx + 1)), 5)
                val_tqdm.set_postfix(val_loss=average_val_loss)
        
        if epoch+1 % 20 == 0 and saveModel:
            torch.save(model, savePath + f'/epoch{epoch+1}-{average_loss}-{average_val_loss}.pt')
    if saveModel:
        torch.save(model.state_dict(), savePath + f'/final.pt')
    else:
        return model

def inference(model, lossFunction, config, dataSets):
    dataLoaderList = [DataLoader(data, batch_size=config.train.batch_size, shuffle=False) for data in dataSets]
    model.eval()
    for i in tqdm(range(len(dataLoaderList)), desc='Inference'):
        dataLoader = dataLoaderList[i]
        dataSet = dataSets[i]
        for batch_idx, batch in enumerate(dataLoader):
            x, y, mainSet_idx = batch
            y = y.to(device)
            x = x.float().to(device)
            if len(x.shape) == 3:
                x = x[:, None, ...]
            output = model(x)
            _, predicted = torch.max(output, 1)
            dataSet.updateLabel(mainSet_idx.cpu().numpy(), predicted.cpu().numpy())

def voteInference(modelList, lossFunction, config, dataSets):
    dataLoaderList = [DataLoader(data, batch_size=config.train.batch_size, shuffle=False) for data in dataSets]
    for modelIdx in range(len(modelList)):
        model = modelList[modelIdx]
        model.eval()
        for i in tqdm(range(len(dataLoaderList)), desc='Inference with model '+str(modelIdx)):
            dataLoader = dataLoaderList[i]
            dataSet = dataSets[i]
            for batch_idx, batch in enumerate(dataLoader):
                x, y, mainSet_idx = batch
                y = y.to(device)
                x = x.float().to(device)
                if len(x.shape) == 3:
                    x = x[:, None, ...]
                output = model(x)
                loss = lossFunction(output, y)
                dataSet.updateVoteRecord(mainSet_idx.cpu().numpy(), 
                                         output.cpu().detach().numpy(),
                                         modelIdx)
    # vote to update pseudo-label
    for dataset in dataSets:
        dataset.voteUpdateLabel()


def semiSupervisedLearning(savePath, device, lossFunction, config, labeledSet, unlabeledSet, valLoader=None, iteration=1):
    # Train model with labeled data
    print('Training with labeled data')
    labeledLoader = DataLoader(labeledSet, batch_size=config.train.batch_size, shuffle=True)
    model = train(savePath, device, lossFunction, config, labeledLoader, valLoader, saveModel=False)

    for i in range(iteration):
        # Update pseudo-label of unlabeled data with model
        print('Inference with unlabeled data')
        inference(model, lossFunction, config, unlabeledSet)
        
        # Train model with labeled and pseudo-labeled data
        print('Training with labeled and pseudo-labeled data')
        mixedData = torch.utils.data.ConcatDataset([labeledSet] + unlabeledSet)
        mixedLoader = DataLoader(mixedData, batch_size=config.train.batch_size, shuffle=True)
        saveModel = True if i == iteration-1 else False
        model = train(savePath, device, lossFunction, config, mixedLoader, valLoader, saveModel=saveModel)
    print('Training completed, model saved at', savePath)

def votedSemiSupervisedLearning(savePath, device, lossFunction, config, labeledSet, unlabeledSet, valLoader=None):
    # Train model with labeled data
    print('Training with labeled data')
    labeledLoader = DataLoader(labeledSet, batch_size=config.train.batch_size, shuffle=True)
    model = train(savePath, device, lossFunction, config, labeledLoader, valLoader, saveModel=False)

    # Update pseudo-label of unlabeled data with model
    print('Inference with unlabeled data')
    inference(model, lossFunction, config, unlabeledSet)

    # Voting module
    # Train model with labeled and pseudo-labeled data (AB, AC, AD,..., AZ)
    print('Bagging with labeled and pseudo-labeled data')
    modelList = []
    for i in range(len(unlabeledSet)):
        print(f'Training with pseudo-labeled data (Model {i})')
        mixedData = torch.utils.data.ConcatDataset([labeledSet] + [unlabeledSet[i]])
        mixedLoader = DataLoader(mixedData, batch_size=config.train.batch_size, shuffle=True)
        model = train(savePath, device, lossFunction, config, mixedLoader, valLoader, saveModel=False)
        modelList.append(model)
    # Inference with test data and vote
    print('Inference with unlabeled data')
    voteInference(modelList, lossFunction, config, unlabeledSet)

    # semi-supervised learning with voted pseudo-label
    print('Training with voted pseudo-labeled data')
    mixedData = torch.utils.data.ConcatDataset([labeledSet] + unlabeledSet)
    mixedLoader = DataLoader(mixedData, batch_size=config.train.batch_size, shuffle=True)
    model = train(savePath, device, lossFunction, config, mixedLoader, valLoader, saveModel=False)
    inference(model, lossFunction, config, unlabeledSet)

    # train final model
    print('Training final model')
    mixedData = torch.utils.data.ConcatDataset([labeledSet] + unlabeledSet)
    mixedLoader = DataLoader(mixedData, batch_size=config.train.batch_size, shuffle=True)
    train(savePath, device, lossFunction, config, mixedLoader, valLoader, saveModel=True)
    print('Training completed, model saved at', savePath)

if __name__ == "__main__":
    configPath = 'configuration/config.yaml'
    labeledSplit = 0.2 # percentage of labeled data
    numGroups = 4   # number of unlabeled data groups
    iteration = 1   # number of iteration for semi-supervised learning
    config = OmegaConf.load(configPath)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # create checkpoint folder
    current_time = datetime.datetime.now()
    file_name = current_time.strftime("%Y-%m-%d_%H%M%S")
    savePath = f'ckpt/{file_name}'
    os.makedirs(savePath)

    # load dataset
    trainData = trainSource(numGroups=numGroups, valSplit=config.train.val_split, labeledSplit=labeledSplit, augmentation=True, dataBalanced=True)
    labeledData = trainData.labeledSet
    unlabeledData = trainData.unlabeledSet

    if config.train.val_split > 0:
        valLoader = DataLoader(trainData.valSet, batch_size=config.train.batch_size, shuffle=True)
    else:
        valLoader = None
    
    lossFunction = nn.CrossEntropyLoss()

    '''
        Trainning mode
            - train model with labeled data
            - semi-supervised learning with labeled and unlabeled data
            - voted semi-supervised learning with labeled and unlabeled data
    '''
    # trainLoader = DataLoader(labeledData, batch_size=config.train.batch_size, shuffle=True)
    # train(savePath, device, lossFunction, config, trainLoader, valLoader=valLoader, saveModel=True)
    # semiSupervisedLearning(savePath, device, lossFunction, config, labeledData, unlabeledData, valLoader, iteration=iteration)
    votedSemiSupervisedLearning(savePath, device, lossFunction, config, labeledData, unlabeledData, valLoader)