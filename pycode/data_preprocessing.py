import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict

class DataPreprocessor:
    # 10초에 5000개. 1초에 500개. 0.5초에 250개. 0.2초는 100개.
    def __init__(self, dirPath, targets=["rubbing", "crumple"], window_size=100, hop_size=50):
        fileNames, labels = self.getFileLists_(dirPath, targets)
        
        dataList = defaultdict(list)
        labelList = defaultdict(list)
        for fileName, label in zip(fileNames, labels):
            dataFile = pd.read_csv(os.path.join(dirPath, fileName))
            dataFile = self.removeTimestamp_(dataFile)
            dataFile = dataFile.values # to the numpy data format
            print(f"  label: [{label}] {fileName}: {dataFile.shape}")

            dataWindowed = self.makeWindowedData_(dataFile, window_size=window_size, hop_size=hop_size)
            indList = self.splitData(len(dataWindowed), ratio=[0.7, 0.1, 0.2], seed=42)
            
            for dataType, ind in indList.items():
                dataList[dataType].append(dataWindowed[ind])
                labelList[dataType].extend([label] * len(ind))
        
        mean=0
        std=0
        for dataType in dataList.keys():
            data = np.vstack(dataList[dataType]).astype(np.int64)
            label = np.array(labelList[dataType]).astype(np.int64)
            if dataType == "train":
                procData = np.log1p(np.abs(data))
                mean = np.mean(procData, axis=(0,1)).astype(np.float64)
                std = np.std(procData, axis=(0,1), ddof=0).astype(np.float64)
                # for the numerical stability
                std[std==0] = 1e-8

            # save the binary data
            with open(os.path.join(dirPath, dataType+".bin"), "wb") as f:
                f.write(data.tobytes() + label.tobytes())
            # save the metadata
            metadata = {
                "shape": list(data.shape),
                "dtype": {
                    "data": str(data.dtype),
                    "label": str(label.dtype)
                },
                "size": data.itemsize,
                "proc": str("abs + log1p"),
                "mean": list(mean),
                "std": list(std)
            }
            with open(os.path.join(dirPath, dataType+"_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=4)
                
        
            # self.dataTrain.append(dataWindowed[trainIdx])
            # self.dataVal.append(dataWindowed[valIdx])
            # self.dataTest.append(dataWindowed[testIdx])
            # self.labelsTrain.extend([label] * len(trainIdx))
            # self.labelsVal.extend([label] * len(valIdx))
            # self.labelsTest.extend([label] * len(testIdx))

        # self.dataTrain = np.vstack(self.dataTrain)
        # self.dataVal = np.vstack(self.dataVal)
        # self.dataTest = np.vstack(self.dataTest)
        # self.labelsTrain = np.array(self.labelsTrain)
        # self.labelsVal = np.array(self.labelsVal)
        # self.labelsTest = np.array(self.labelsTest)

        # print(self.dataTrain.shape, self.dataVal.shape, self.dataTest.shape)
        # print(self.labelsTrain.shape, self.labelsVal.shape, self.labelsTest.shape)

        # # log + standard normalization
        # self.dataTrain = np.log1p(np.abs(self.dataTrain))
        # self.dataVal = np.log1p(np.abs(self.dataVal))
        # self.dataTest = np.log1p(np.abs(self.dataTest))

        # self.mean = np.mean(self.dataTrain, axis=(0,1))
        # self.std = np.std(self.dataTrain, axis=(0,1), ddof=0)
        # self.std[self.std == 0] = 1e-8  # 안정성
        # print(f"Data mean: {self.mean.shape}, std: {self.std.shape}")

        # self.dataTrain = (self.dataTrain - self.mean) / self.std
        # self.dataVal = (self.dataVal - self.mean) / self.std
        # self.dataTest = (self.dataTest - self.mean) / self.std

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
    dataPrep = DataPreprocessor("./data/raw_data_merged/", targets=["idle", "rubbing", "crumple"])