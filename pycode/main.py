from dataset import DataPreprocessor, FFTDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import FFT2DCNN

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
plt.rcParams.update({
    "font.size": 16,        # 기본 폰트 크기
    "axes.titlesize": 16,   # 제목 크기
    "axes.labelsize": 16,   # 축 제목 크기
    "xtick.labelsize": 16,  # x축 글자 크기
    "ytick.labelsize": 16   # y축 글자 크기
})

def main():
    targets = ["idle", "rubbing", "crumple"]
    window_size = 200
    hop_size = 100
    dataPrep = DataPreprocessor("./data/merged_fft_data/", targets=targets, window_size=window_size, hop_size=hop_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    datasetTrain = FFTDataset(dataPrep.dataTrain, dataPrep.labelsTrain, device=device, dtype="train")
    datasetVal = FFTDataset(dataPrep.dataVal, dataPrep.labelsVal, device=device)
    datasetTest = FFTDataset(dataPrep.dataTest, dataPrep.labelsTest, device=device)

    dataLoaderTrain = DataLoader(datasetTrain, batch_size=32, shuffle=True, drop_last=True)
    dataLoaderVal = DataLoader(datasetVal, batch_size=32, shuffle=False, drop_last=False)
    dataLoaderTest = DataLoader(datasetTest, batch_size=32, shuffle=False, drop_last=False)


    model = FFT2DCNN(n_classes=3).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    bestAcc = 0
    bestModel = None

    step = 0
    dataIter = iter(dataLoaderTrain)
    while step < 10000:
        step += 1
        try:
            X, Y = next(dataIter)
        except StopIteration:
            dataIter = iter(dataLoaderTrain)

        model.train()
        optimizer.zero_grad()
        y = model(X)
        loss = criterion(y, Y)
        loss.backward()
        optimizer.step()


        if step % 100 == 0:
            loss = 0
            correct = 0
            with torch.no_grad():
                model.eval()
                for XVal, YVal in dataLoaderVal:
                    pred = model(XVal)
                    loss += criterion(pred, YVal).item() * YVal.size(0)

                    pred = torch.argmax(pred, dim=1)
                    correct += (pred == YVal).sum().item()
                    
            acc = correct / len(datasetVal)
            loss = loss / len(datasetVal)
            print(f"[Step {step}] Loss: {loss:.4f} Acc: {acc:.4f}")
            if acc > bestAcc:
                bestAcc = acc
                bestModel = model.state_dict()
    
    print(f"Best Acc: {bestAcc:.4f}")
    model.load_state_dict(bestModel)
    model.eval()
    
    lossTotal = 0
    correct = 0
    samples = 0
    preds = []
    labels = []
    with torch.no_grad():
        for XTest, YTest in dataLoaderTest:
            pred = model(XTest)
            loss = criterion(pred, YTest)
            
            lossTotal += loss.item() * YTest.size(0)

            pred = torch.argmax(pred, dim=1)
            preds.append(pred.cpu())
            labels.append(YTest.cpu())
            

            correct += (pred == YTest).sum().item()
            samples += YTest.size(0)


    lossAvg = lossTotal / samples
    acc = correct / samples
    print(f"Test Loss: {lossAvg:.4f} Acc: {acc:.4f}")       


    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:\n", cm)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=targets, yticklabels=targets, cbar=True, annot_kws={"size": 16})

    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    params = model.state_dict()
    torch.save(params, "./output/model.prm")



if __name__ == "__main__":
    main()