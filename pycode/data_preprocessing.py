import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict

class DataPreprocessor:
    # 10초에 5000개. 1초에 500개. 0.5초에 250개. 0.2초는 100개.
    def __init__(self, dirPath, targets=["rubbing.csv", "crumple.csv"]):
        self.files, self.labels = self.getFileLists_(dirPath, targets)
        self.dataType = None

    def initData_(self):
        self.dataList = defaultdict(list)
        self.labelList = defaultdict(list)

        self.dataTrain = []
        self.dataVal = []
        self.dataTest = []

        self.labelTrain = []
        self.labelVal = []
        self.labelTest = []

        self.windowSize=None
        self.hopSize=None

        self.mean=0
        self.std=0

    def getTrainData(self):
        data = self.dataTrain
        label = self.labelTrain
        data = np.log1p(np.abs(self.dataTrain))
        data = (data - self.mean) / self.std
        return data, label
    
    def getValData(self):
        self.dataType = "val"
        data = self.dataList["val"]
        label = self.labelList["val"]
        data = np.log1p(np.abs(data))
        data = (data - self.mean) / self.std
        return self

    def getTestData(self):
        self.dataType = "test"
        data = self.dataList["test"]
        label = self.labelList["test"]
        data = np.log1p(np.abs(data))
        data = (data - self.mean) / self.std
        return self

    def saveData(self, outPath):
        os.makedirs(outPath, exist_ok=True)
        # save the binary data
        with open(os.path.join(outPath, "train.bin"), "wb") as f:
            f.write(self.dataTrain.tobytes() + self.labelTrain.tobytes())
        with open(os.path.join(outPath, "val.bin"), "wb") as f:
            f.write(self.dataVal.tobytes() + self.labelVal.tobytes())
        with open(os.path.join(outPath, "test.bin"), "wb") as f:
            f.write(self.dataTest.tobytes() + self.labelTest.tobytes())
        print("Save the binary files.")

        # save the metadata
        metadata = {
            "windowSize": int(self.windowSize),
            "hopSize": int(self.hopSize),
            "shape": {
                "train": tuple(self.dataTrain.shape),
                "val": tuple(self.dataVal.shape),
                "test": tuple(self.dataTest.shape)
            },
            "dtype": {
                "data": str(self.dataTrain.dtype),
                "label": str(self.labelTrain.dtype)
            },
            "itemSize": int(self.dataTrain.itemsize),
            "proc": tuple(["abs", "log1p"]),
            "mean": list(self.mean),
            "std": list(self.std)
        }
        with open(os.path.join(outPath, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)
        print("Save the metadata.")
        
    def setWindowAndHopSize(self, window_size=100, hop_size=50):
        self.initData_()
        self.windowSize = window_size
        self.hopSize = hop_size
        print(f"Window size: {window_size}, Hop size: {hop_size}")

        for data, label in zip(self.files, self.labels):
            data = self.removeTimestamp_(data)
            data = data.values # to the numpy data format

            dataWindowed = self.makeWindowedData_(data, window_size=window_size, hop_size=hop_size)
            idxTrain, idxVal, idxTest = self.splitData(len(dataWindowed), ratio=[0.7, 0.1, 0.2], seed=42)
            self.dataTrain.append(dataWindowed[idxTrain])
            self.dataVal.append(dataWindowed[idxVal])
            self.dataTest.append(dataWindowed[idxTest])

            self.labelTrain.append([label] * len(idxTrain))
            self.labelVal.append([label] * len(idxVal))
            self.labelTest.append([label] * len(idxTest))

               
        self.dataTrain = np.vstack(self.dataTrain).astype(np.int64)
        self.dataVal = np.vstack(self.dataVal).astype(np.int64)
        self.dataTest = np.vstack(self.dataTest).astype(np.int64)

        self.labelTrain = np.hstack(self.labelTrain).astype(np.int64)
        self.labelVal = np.hstack(self.labelVal).astype(np.int64)
        self.labelTest = np.hstack(self.labelTest).astype(np.int64)
        # Calculate the mean, and standard deviation.
        dataProc = np.log1p(np.abs(self.dataTrain))
        self.mean = np.mean(dataProc, axis=(0,1)).astype(np.float64)
        self.std = np.std(dataProc, axis=(0,1), ddof=0).astype(np.float64)

        # self.dataTrain = (self.dataTrain - self.mean) / self.std
        # self.dataVal = (self.dataVal - self.mean) / self.std
        # self.dataTest = (self.dataTest - self.mean) / self.std
        # for the numerical stability
        # self.std[self.std < 1e-8] = 1e-8

    def getFileLists_(self, dirPath, targets):
        targetFiles = os.listdir(dirPath)

        files = []
        labels = []
        for target in targetFiles:
            if target in targets:
                print(f"Found [{target}] file.")
                file = pd.read_csv(os.path.join(dirPath, target))
                print(f"  Shape: {file.shape}.")
                label = targets.index(target)
                print(f"  Index: {label}")
                files.append(file)
                labels.append(label)

        return files, labels

        
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
    
    def splitData(self, N, ratio=[0.7, 0.1, 0.2], seed=42): # ratio=[train, val, test]
        assert sum(ratio) == 1.0, "Sum of ratio must be 1.0"

        indices = np.arange(N)
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        train_end = int(N * ratio[0])
        val_end = int(N * (ratio[0] + ratio[1]))

        idxTrain = indices[:train_end]
        idxVal = indices[train_end:val_end]
        idxTest = indices[val_end:]
        print(f"Expected data length: {len(idxTrain)}(train), {len(idxVal)}(val), {len(idxTest)}(test)")
        return idxTrain, idxVal, idxTest


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # from dataset import FFTDataset

    dataPrep = DataPreprocessor("./data/raw_data_merged/", targets=["idle.csv", "rubbing.csv", "crumple.csv"])
    dataPrep.setWindowAndHopSize(100, 25)
    # dataPrep.setWindowAndHopSize(50, 25)
    # dataPrep.saveData("./data/train3/")

    datasetTrain = FFTDataset(dataPrep.dataTrain, dataPrep.labelTrain, )

    # dataLoaderTrain = DataLoader(datasetTrain, batch_size=32, shuffle=True)

    # dataLoaderTrain = DataLoader(dataPrep.getTrainData, batch_size=32, shuffle=True, drop_last=True)

    # print(len(dataLoaderTrain))
    # dataIter = iter(dataLoaderTrain)
    # while 1:
    #     try:
    #         data, labels = next(dataIter)
    #     except StopIteration:
    #         dataIter = iter(dataLoaderTrain)
    #     print(data.shape, labels)

