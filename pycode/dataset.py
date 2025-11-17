import pandas as pd
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import torch

class DataPreprocessor:
    # 10초에 5000개. 1초에 500개. 0.5초에 250개. 0.2초는 100개.
    def __init__(self, dirPath, targets=["rubbing", "crumple"], window_size=100, hop_size=50):
        fileNames, labels = self.getFileLists_(dirPath, targets)
        
        self.dataTrain = []
        self.dataVal = []
        self.dataTest = []
        self.labelsTrain = []
        self.labelsVal = []
        self.labelsTest = []
        for fileName, label in zip(fileNames, labels):
            dataFile = pd.read_csv(os.path.join(dirPath, fileName))
            dataFile = self.removeTimestamp_(dataFile)
            dataFile = dataFile.values
            print(f"  label: [{label}] {fileName}: {dataFile.shape}")

            dataWindowed = self.makeWindowedData_(dataFile, window_size=window_size, hop_size=hop_size)
            trainIdx, valIdx, testIdx = self.splitData(dataWindowed, ratio=[0.7, 0.1, 0.2], seed=42)
            self.dataTrain.append(dataWindowed[trainIdx])
            self.dataVal.append(dataWindowed[valIdx])
            self.dataTest.append(dataWindowed[testIdx])
            self.labelsTrain.extend([label] * len(trainIdx))
            self.labelsVal.extend([label] * len(valIdx))
            self.labelsTest.extend([label] * len(testIdx))

        self.dataTrain = np.vstack(self.dataTrain)
        self.dataVal = np.vstack(self.dataVal)
        self.dataTest = np.vstack(self.dataTest)
        self.labelsTrain = np.array(self.labelsTrain)
        self.labelsVal = np.array(self.labelsVal)
        self.labelsTest = np.array(self.labelsTest)

        print(self.dataTrain.shape, self.dataVal.shape, self.dataTest.shape)
        print(self.labelsTrain.shape, self.labelsVal.shape, self.labelsTest.shape)

        # log + standard normalization
        self.dataTrain = np.log1p(np.abs(self.dataTrain))
        self.dataVal = np.log1p(np.abs(self.dataVal))
        self.dataTest = np.log1p(np.abs(self.dataTest))

        self.mean = np.mean(self.dataTrain, axis=(0,1))
        self.std = np.std(self.dataTrain, axis=(0,1), ddof=0)
        self.std[self.std == 0] = 1e-8  # 안정성
        print(f"Data mean: {self.mean.shape}, std: {self.std.shape}")


        self.dataTrain = (self.dataTrain - self.mean) / self.std
        self.dataVal = (self.dataVal - self.mean) / self.std
        self.dataTest = (self.dataTest - self.mean) / self.std


    def getFileLists_(self, dirPath, targets=["rubbing", "crumple"]):
        files = os.listdir(dirPath)
        fileNames = []
        labels = []
        print(f"Find {len(files)} files({', '.join(files)}) in {dirPath}")

        for file in files:
            if any(t in file for t in targets):
                label = targets.index(next((t for t in targets if t in file), None))
                fileNames.append(file)
                labels.append(label)
        return fileNames, labels
        
    def removeTimestamp_(self, data):
        return data.iloc[:, 1:]
    

    def makeWindowedData_(self, data, window_size=200, hop_size=100):
        ret = []
        for start in range(0, (len(data) - window_size) + 1, hop_size):
            ret.append(data[start:start + window_size])
        print(f"    Total windows created: {len(ret)}")
        return np.array(ret)

    def makeWindowedDataWithLabel_(self, data, label, window_size=200, hop_size=100):
        dataWindowed = self.makeWindowedData_(data, window_size, hop_size)
        return [(window, label) for window in dataWindowed]
    
    def splitData(self, data, ratio=[0.7, 0.1, 0.2], seed=42):
        assert sum(ratio) == 1.0, "Sum of ratio must be 1.0"
        n = len(data)
        indices = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        train_end = int(n * ratio[0])
        val_end = int(n * (ratio[0] + ratio[1]))

        trainIdx = indices[:train_end]
        valIdx = indices[train_end:val_end]
        testIdx = indices[val_end:]
        return trainIdx, valIdx, testIdx



class FFTDataset(Dataset):
    def __init__(self, data, labels, dtype=None, device=torch.device("cpu"), transforms=None):
        self.data = data
        self.labels = labels
        self.dtype = dtype
        self.device = device
        self.transforms = transforms.Compose(transforms) if transforms is not None else None

    def __len__(self):
        if self.dtype == "train":
            return 100000000
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        idx = idx % len(self.data)
        x = self.data[idx]
        y = self.labels[idx]
        if self.transforms:
            x = self.transforms(x)
        return torch.tensor(x, dtype=torch.float32, device=self.device), \
                torch.tensor(y, dtype=torch.long, device=self.device)


def main():
    dataPrep = DataPreprocessor("./data/merged_fft_data/")
    datasetTrain = FFTDataset(dataPrep.dataTrain, dataPrep.labelsTrain, dtype="train")
    datasetVal = FFTDataset(dataPrep.dataVal, dataPrep.labelsVal)
    datasetTest = FFTDataset(dataPrep.dataTest, dataPrep.labelsTest)

    dataLoaderTrain = DataLoader(datasetTrain, batch_size=32, shuffle=True)
    dataLoaderVal = DataLoader(datasetVal, batch_size=32, shuffle=False)
    dataLoaderTest = DataLoader(datasetTest, batch_size=32, shuffle=False)

    # print(len(dataLoaderTrain))
    dataIter = iter(dataLoaderTrain)
    while 1:
        try:
            data, labels = next(dataIter)
        except StopIteration:
            dataIter = iter(dataLoaderTrain)
        print(data.shape, labels)

if __name__ == "__main__":
    main()