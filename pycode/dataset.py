import pandas as pd
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch
import json
from data_preprocessing import DataPreprocessor

class FFTDataset(Dataset):
    def __init__(self, dataPrep:DataPreprocessor, device=torch.device("cpu"), transforms=None):
        self.dataPrep = dataPrep
        self.device = device
        self.transforms = transforms.Compose(transforms) if transforms is not None else None

        # with open(metadataPath, 'r') as f:
        #     metadata = json.load(f)

        # self.type = metadata["type"]
        # self.dataShape = metadata["shape"]
        # self.dtypeData = metadata["dtype"]["data"]
        # self.dtypeLabel = metadata["dtype"]["label"]
        # self.itemSize = metadata["size"]

        # dataLength = np.prod(self.dataShape) * self.itemSize
        # with open(dataPath, 'rb') as f:
        #     dataBin = f.read()

        # dataByte = dataBin[:dataLength]
        # labelByte = dataBin[dataLength:]

        # self.data = np.frombuffer(dataByte, dtype=np.dtype(self.dtypeData)).reshape(self.dataShape)
        # self.label = np.frombuffer(labelByte, dtype=np.dtype(self.dtypeLabel)).reshape(self.dataShape[0])

        # proc = metadata["proc"]
        # mean = metadata["mean"]
        # std = metadata["std"]
        print(f"{dataPrep.dataType} is loaded. {dataPrep.}")
        
        self.data, self.label = dataPrep.getTrainData()
        self.data = np.log1p(np.abs(self.data))
        self.data = (self.data - dataPrep.mean) / dataPrep.std

    def __len__(self):
        if self.dataPrep.dataType == "train":
            return 100000000
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        idx = idx % len(self.data)
        x = self.data[idx]
        y = self.label[idx]
        if self.transforms:
            x = self.transforms(x)
        return torch.tensor(x, dtype=torch.float32, device=self.device), \
                torch.tensor(y, dtype=torch.long, device=self.device)


def main():

    dataPrep = DataPreprocessor("./data/raw_data_merged/", targets=["idle", "rubbing", "crumple"])
    dataPrep.setWindowAndHopSize(100, 50)
    
    datasetTrain = FFTDataset(dataPrep)
    # datasetVal = FFTDataset(dataPrep.dataVal, dataPrep.labelsVal)
    # datasetTest = FFTDataset(dataPrep.dataTest, dataPrep.labelsTest)

    dataLoaderTrain = DataLoader(datasetTrain, batch_size=32, shuffle=True)
    # dataLoaderVal = DataLoader(datasetVal, batch_size=32, shuffle=False)
    # dataLoaderTest = DataLoader(datasetTest, batch_size=32, shuffle=False)

    print(len(dataLoaderTrain))
    dataIter = iter(dataLoaderTrain)
    while 1:
        try:
            data, labels = next(dataIter)
        except StopIteration:
            dataIter = iter(dataLoaderTrain)
        print(data.shape, labels)

if __name__ == "__main__":
    main()