import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict

class DataPreprocessor:
    # 10초에 5000개. 1초에 500개. 0.5초에 250개. 0.2초는 100개.
    def __init__(self, targetPath, targets=["rubbing", "crumple"]):
        self.fileNames, self.labels = self.getFileLists_(targetPath, targets)
        self.dataType = None

    def initData_(self):
        self.dataList = defaultdict(list)
        self.labelList = defaultdict(list)

        self.windowSize=None
        self.hopSize=None

        self.mean=0
        self.std=0

    def getTrainData(self):
        self.dataType = "train"
        data = self.dataList["train"]
        label = self.labelList["train"]
        data = np.log1p(np.abs(data))
        data = (data - self.mean) / self.std
        return self
    
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
        for dataType, data, label in zip(self.dataList.keys(), self.dataList.values(), self.labelList.values()):
            with open(os.path.join(outPath, dataType+".bin"), "wb") as f:
                f.write(data.tobytes() + label.tobytes())
            # save the metadata
            metadata = {
                "type" : str(dataType),
                "windowSize": int(self.windowSize),
                "hopSize": int(self.hopSize),
                "shape": list(data.shape),
                "dtype": {
                    "data": str(data.dtype),
                    "label": str(label.dtype)
                },
                "size": data.itemsize,
                "proc": tuple(["abs", "log1p"]),
                "mean": list(self.mean),
                "std": list(self.std)
            }
            with open(os.path.join(outPath, dataType+".json"), "w") as f:
                    json.dump(metadata, f, indent=4)
                    

    def setWindowAndHopSize(self, window_size=100, hop_size=50):
        self.initData_()
        self.windowSize = window_size
        self.hopSize = hop_size

        for fileName, label in zip(self.fileNames, self.labels):
            data = pd.read_csv(fileName)
            data = self.removeTimestamp_(data)
            data = data.values # to the numpy data format
            print(f"  label: [{label}] {fileName}: {data.shape}")

            dataWindowed = self.makeWindowedData_(data, window_size=window_size, hop_size=hop_size)
            indList = self.splitData(len(dataWindowed), ratio=[0.7, 0.1, 0.2], seed=42)
            
            for dataType, ind in indList.items():
                self.dataList[dataType].append(dataWindowed[ind])
                self.labelList[dataType].extend([label] * len(ind))
        
        for dataType in self.dataList.keys():
            self.dataList[dataType] = np.vstack(self.dataList[dataType]).astype(np.int64)
            self.labelList[dataType] = np.array(self.labelList[dataType]).astype(np.int64)

            if dataType == "train":
                procedData = self.procData(self.dataList[dataType])
                self.mean = np.mean(procedData, axis=(0,1)).astype(np.float64)
                self.std = np.std(procedData, axis=(0,1), ddof=0).astype(np.float64)
                # for the numerical stability
                # self.std[self.std < 1e-8] = 1e-8

    def procData(self, x):
        y = np.log1p(np.abs(x))
        return y

    def getFileLists_(self, dirPath, targets=["rubbing", "crumple"]):
        files = os.listdir(dirPath)
        fileNames = []
        labels = []
        print(f"Find {len(files)} files({', '.join(files)}) in {dirPath}")

        for file in files:
            if any(t in file for t in targets):
                label = targets.index(next((t for t in targets if t in file), None))
                fileNames.append(os.path.join(dirPath, file))
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
        print(f"Expected data length: {len(idxTrain)}, {len(idxVal)}, {len(idxTest)}")
        return {"train":idxTrain, "val":idxVal, "test":idxTest}


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataPrep = DataPreprocessor("./data/raw_data_merged/", targets=["idle", "rubbing", "crumple"])
    dataPrep.setWindowAndHopSize(100, 50)

    datasetTrain = FFTDataset("")

    dataLoaderTrain = DataLoader(datasetTrain, batch_size=32, shuffle=True)

    dataLoaderTrain = DataLoader(dataPrep.getTrainData, batch_size=32, shuffle=True, drop_last=True)

    print(len(dataLoaderTrain))
    dataIter = iter(dataLoaderTrain)
    while 1:
        try:
            data, labels = next(dataIter)
        except StopIteration:
            dataIter = iter(dataLoaderTrain)
        print(data.shape, labels)

