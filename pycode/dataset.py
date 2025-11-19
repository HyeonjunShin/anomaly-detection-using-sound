import pandas as pd
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import torch


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