import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

BASE_PATH = "./data/anomaly_detection_sound_data/csv_origin/"

temp_data = []

def main():
    paths = os.listdir(BASE_PATH)
    for path in paths[:1]:
        splited_name = path.split(" ")
        csv_file = pd.read_csv(BASE_PATH + path, header=None)
        length = csv_file.shape[0]
        hole_size = splited_name[8][:-1]

        if splited_name[1] == "성공_":
            new_col_data = [1] * length
            # print(hole_size)
        elif splited_name[1] == "실패_":
            new_col_data = [0] * length
            print(hole_size)
        print(csv_file.iloc[:1,7:39])
        csv_file.iloc[:2,7:39].plot()
        plt.show()



    #     # csv_file.
    #     # csv_file.insert(loc=csv_file.columns, value=new_col_data)
    #     # print(splited_name[1], path.split(" ")[8])

    #     # print(csv_file.shape[0], len(csv_file))
    #     print(csv_file.iloc[:,7:39])
    #     # print(csv_file.shape)
    #     csv_file[csv_file.shape[1]] = hole_size
    #     csv_file[csv_file.shape[1]] = new_col_data
    #     # print(csv_file.shape)

    #     temp_data.append(csv_file)

    # NEW_DATA = pd.concat(temp_data)
    # print(NEW_DATA.shape)
    # # print(NEW_DATA.iloc[:,7:39])


if __name__ == "__main__":
    main()
