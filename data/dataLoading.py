from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import numpy as np
import random

class testSource(Dataset):
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.data = self.dataset.data
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
        self.labels = self.dataset.targets
        if not isinstance(self.labels, torch.Tensor):
            self.labels = torch.tensor(self.labels)
        self.mainset_idx = list(range(len(self.data)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, label, mainset_idx = self.data[index], self.labels[index], self.mainset_idx[index]
        if len(data.shape) == 2:
            data = data[:, :, None]
        data = data.transpose(2, 0, 1)
        data = torch.from_numpy(data).float()
        label = torch.tensor(label)
        return data, label, mainset_idx

class trainSubset(Dataset):
    def __init__(self, data, groundTruth, mainset_idx, labeled=True, augmentation=False):
        self.data = data
        self.groundTruth = groundTruth
        self.mainset_idx = mainset_idx
        self.mainSet2Subset = {mainset_idx[i]: i for i in range(len(mainset_idx))}
        self.augmentation = augmentation
        self.voteRecord = []
        if labeled:
            self.labels = self.groundTruth
        else:
            self.labels = np.ones(len(self.groundTruth)) * -1
    
    def updateLabel(self, idx, label):
        for i in range(len(idx)):
            self.labels[self.mainSet2Subset[idx[i]]] = label[i]
    
    def updateVoteRecord(self, idx, vote, modelIdx):
        if len(self.voteRecord) == modelIdx:
            self.voteRecord.append([-1] * self.data.shape[0])
        for i in range(len(idx)):
            self.voteRecord[modelIdx][self.mainSet2Subset[idx[i]]] = vote[i]
    
    def voteUpdateLabel(self):
        numGuess = len(self.voteRecord)
        matrix = np.array(self.voteRecord)
        matrix = np.sum(matrix, axis=0) / numGuess
        self.labels = np.argmax(matrix, axis=1)
        self.voteRecord = []

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, label, mainset_idx = self.data[index], self.labels[index], self.mainset_idx[index]
        if self.augmentation:
            data = data.copy()
            if len(data.shape) == 2:
                height, width = data.shape
                channel = 1
                data = data[:, :, None]
            else:
                height, width, channel = data.shape
            if np.random.rand() < 0.5:
                data = np.flip(data, axis=1)
            # shift the colored image
            if np.random.rand() < 0.5:
                dx, dy = np.random.randint(-2, 3, 2)
                data =  cv2.warpAffine(data, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height))
                if channel == 1:
                    data = data[:, :, None]
            # rotate the image
            if np.random.rand() < 0.5:
                angle = np.random.randint(-15, 16)
                M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
                data = cv2.warpAffine(data, M, (width, height))
                if channel == 1:
                    data = data[:, :, None]
            # change the brightness
            data = cv2.convertScaleAbs(data, 
                                       alpha=np.random.uniform(0.9, 1.1),
                                       beta=np.random.randint(-10, 11))
            if channel == 1:
                data = data[:, :, None]
        data = data.transpose(2, 0, 1)
        data = torch.from_numpy(data).float()
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label).to(torch.long)
        return data, label, mainset_idx # idx allow keep track of the loss value of each sample

'''
trainSource init arguments:
    numGroups: number of unlabeled groups
    valSplit: the proportion of validation set
    augmentation: whether to use data augmentation

    This is to mimic the semi-supervised learning scenario (for numGroup > 1)
'''
class trainSource(Dataset):
    def __init__(self, numGroups=5, valSplit=0, labeledSplit = 0.2, augmentation=False, dataBalanced=False):
        # Load the MNIST dataset
        transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        #self.dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
        self.dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.valSplit = valSplit
        self.labeledSplit = labeledSplit
        self.data = self.dataset.data
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
        self.labels = self.dataset.targets
        if not isinstance(self.labels, torch.Tensor):
            self.labels = torch.tensor(self.labels)
        self.numGroups = numGroups
        self.augmentation = augmentation
        self.dataBalanced = dataBalanced
        self.create_groups()

    # dataset.groups[0] is the first group containing [data, labels]
    def create_groups(self):
        if self.dataBalanced:
            # group data idx by class
            classIdx = {}
            for i in range(len(self.labels)):
                label = self.labels[i].item()
                if label not in classIdx:
                    classIdx[label] = []
                classIdx[label].append(i)
            for i in range(10):
                random.Random().shuffle(classIdx[i])
            # split the valSet equally by class
            if self.valSplit > 0:
                valSize = int(len(self.data) * self.valSplit)
                valIndices = []
                for i in range(10):
                    valIndices += classIdx[i][:valSize // 10]
                    classIdx[i] = classIdx[i][valSize // 10:]
                valData = self.data[valIndices]
                valLabels = self.labels[valIndices]
                valSet = trainSubset(valData, valLabels, valIndices)
                self.valSet = valSet
                self.valSetIdx = valIndices
            # compute remaining data size
            dataSize = 0
            for i in classIdx.values():
                dataSize += len(i)
            # split labeled data
            labeledSize = int(dataSize * self.labeledSplit)
            classSize = labeledSize // 10
            group_indices = []
            for i in range(10):
                group_indices += classIdx[i][:classSize]
                classIdx[i] = classIdx[i][classSize:]
            group_data = self.data[group_indices]
            group_labels = self.labels[group_indices]
            labeledSet = trainSubset(group_data, group_labels, group_indices, labeled=True, augmentation=self.augmentation)
            self.labeledSet = labeledSet
            self.labeledSetIdx = group_indices

            if self.numGroups > 0:
                # compute remaining data size
                dataSize = 0
                for i in classIdx.values():
                    dataSize += len(i)
                # split unlabeled data
                groupSize = dataSize // self.numGroups
                self.unlabeledSet = []
                self.unlabeledSetIdx = {}
                for i in range(self.numGroups):
                    classSize = groupSize // 10
                    group_indices = []
                    if i == self.numGroups - 1:
                        for j in range(10):
                            group_indices += classIdx[j]
                    else:
                        for j in range(10):
                            group_indices += classIdx[j][:classSize]
                            classIdx[j] = classIdx[j][classSize:]
                    group_data = self.data[group_indices]
                    group_labels = self.labels[group_indices]
                    subset = trainSubset(group_data, group_labels, group_indices, labeled=False, augmentation=self.augmentation)
                    self.unlabeledSet.append(subset)
                    self.unlabeledSetIdx[i] = group_indices
            else:
                self.unlabeledSet = []
                self.unlabeledSetIdx = {}
        else:
            allIdx = list(range(len(self.data)))
            random.Random(1).shuffle(allIdx) # shuffle the indices
            if self.valSplit > 0:
                valSize = int(len(self.data) * self.valSplit)
                valIndices = allIdx[:valSize]
                valData = self.data[valIndices]
                valLabels = self.labels[valIndices]
                valSet = trainSubset(valData, valLabels, valIndices)
                self.valSet = valSet
                self.valSetIdx = valIndices
                allIdx = allIdx[valSize:]

            # labeled data
            labeledSize = int(len(allIdx) * self.labeledSplit)
            labeledIndices = allIdx[:labeledSize]
            allIdx = allIdx[labeledSize:]
            labeledData = self.data[labeledIndices]
            labeledLabels = self.labels[labeledIndices]
            labeledSet = trainSubset(labeledData, labeledLabels, labeledIndices, labeled=True, augmentation=self.augmentation)
            self.labeledSet = labeledSet
            self.labeledSetIdx = labeledIndices
            # unlabeled data
            if self.numGroups > 0:
                groupSize = len(allIdx) // self.numGroups
                self.unlabeledSet = []
                self.unlabeledSetIdx = {}
                for i in range(self.numGroups):
                    group_indices = allIdx[:groupSize]
                    allIdx = allIdx[groupSize:]
                    group_data = self.data[group_indices]
                    group_labels = self.labels[group_indices]
                    subset = trainSubset(group_data, group_labels, group_indices, labeled=False, augmentation=self.augmentation)
                    self.unlabeledSet.append(subset)
                    self.unlabeledSetIdx[i] = group_indices
            else:
                self.unlabeledSet = []
                self.unlabeledSetIdx = {}

# testing
if __name__ == "__main__":
    testSource = testSource()
    dataset = trainSource(5, valSplit=0.1, augmentation=True, dataBalanced=True)
    valSet = dataset.valSet
    labeledSet = dataset.labeledSet
    unlabeledSet = dataset.unlabeledSet # list of unlabeled set

    labeledLoader = DataLoader(labeledSet, batch_size=32, shuffle=True)
    mixedData = torch.utils.data.ConcatDataset([labeledSet] + unlabeledSet)
    mixedLoader = DataLoader(dataset=mixedData, batch_size=32, shuffle=True)
    for batch in mixedLoader:
        data, label, mainset_idx = batch
        print(data.shape, label, mainset_idx)