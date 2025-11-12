import pandas as pd
import os

def readData(file_path):
    return pd.read_csv(file_path)

def makeWindowedData(data, window_size=10, hop_size=5):
    ret = []
    for start in range(0, len(data) - window_size + 1, hop_size):
        print(start, start + window_size)

def main():
    filePaths = [
        "./data/fft_data/2025-11-11_07-38-32(kim,rubbing,new device).csv",
        "./data/fft_data/2025-11-11_07-41-44(kim,crumple,new device).csv",
        "./data/fft_data/2025-11-11_07-46-25(shin,rubbing,new device).csv",
        "./data/fft_data/2025-11-11_10-58-11(shin,crumple,new device).csv",
        "./data/fft_data/2025-11-12_05-24-36(jung,rubbing,new device).csv",
        "./data/fft_data/2025-11-12_05-27-27(jung,crumple,new device).csv",
        # "./data/fft_data/2025-11-11_11-04-25(shin,idle, new deivce).csv"
    ]
    
    dataDict = {}
    for path in filePaths:
        data = readData(path)
        data = data.iloc[-60000:]
        if "rubbing" in path:
            label = "rubbing"
        elif "crumple" in path:
            label = "crumple"
        
        try:
            dataDict[label].append(data)
        except KeyError:
            dataDict[label] = [data]
    

    for label, data in dataDict.items():
        combinedData = pd.concat(data, ignore_index=True)
        dataDict[label] = combinedData
        print(f"{label}: {combinedData.shape}")
    
    # makeWindowedData(dataDict["rubbing"], window_size=200, hop_size=100)

    for start in range(0, len(range(1000)) - 100 + 1, 100):
        print(start, start + 100)

    


    # dataRubbing = []
    # dataCrumple = []
    # for path in filePaths:
    #     if "rubbing" in path:
    #         data = readData(path)
    #         dataRubbing.append(data)
    #     elif "crumple" in path:
    #         data = readData(path)
    #         dataCrumple.append(data)

    # print(len(dataRubbing), len(dataCrumple))
    # dataRubbingAll = pd.concat(dataRubbing, ignore_index=True)
    # dataCrumpleAll = pd.concat(dataCrumple, ignore_index=True)
    # print(dataRubbingAll.shape, dataCrumpleAll.shape)



if __name__ == "__main__":
    main()