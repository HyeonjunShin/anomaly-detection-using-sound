import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR #lr scheduler


torch.set_default_tensor_type('torch.FloatTensor')

# 하이퍼파라미터
# batch_size = 50
batch_size = 10
class_num = 4
learning_rate = 0.001
num_epochs = 600
kernel_size = 5
num_input = 30
fig_name = '2025-0613_3005_mic_31ch_epoch200_suctionclass_adam_3_batch10'

use_vacuum = False  # vacuum 데이터 사용 여부
use_mic_data = True   # mic_data 사용 여부


train_accuracy_list = []
val_accuracy_list = []

gerneral_path = "/home/lds/workspace/2024-11 suction classification/2024-1127 데이터/3005_suction_class/"


# 1D CNN 모델 정의
class LDS1DCNN(nn.Module):
    def __init__(self, input_size, num_classes, in_channels):
        super(LDS1DCNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(in_channels, in_channels * 18, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels * 18, in_channels * 36, kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels * 36, in_channels * 36, kernel_size, stride=1, padding=1)
        self.fc1 = nn.Linear(36 * math.floor(input_size/2) * in_channels, 256)

        # self.fc1 = nn.Linear(540,256) # kernel 3 vac
        self.fc1 = nn.Linear(12276,256)# kernel 5 mic
        # self.fc1 = nn.Linear(17280,256)# kernel 3 mic+vac
        # self.fc1 = nn.Linear(4212, 256) # mic 3채널


        self.fc2 = nn.Linear(256, num_classes)

        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.drop(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        # print(x.shape)
        x = self.drop(x)
        x = F.leaky_relu(self.fc1(x.flatten(start_dim=1)))
        # print(x.shape)
        # exit()
        x = self.drop(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x

# Xavier initialization
def xavier_init(m):
    
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)

# 사용자 정의 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, root_dir, use_vacuum=True, use_mic_data=True):
        self.root_dir = root_dir
        self.use_vacuum = use_vacuum
        self.use_mic_data = use_mic_data
        self.file_list = self.get_file_list()

        # 채널 수 자동 설정
        self.channel_num = 0
        if self.use_mic_data:
            self.channel_num += 31  # 마이크 데이터 채널

        if self.use_vacuum:
            self.channel_num += 1  # Vacuum 데이터 채널

    def get_file_list(self):
        file_list = []
        for label in os.listdir(self.root_dir):
            label_path = os.path.join(self.root_dir, label)
            if os.path.isdir(label_path):
                for file_name in os.listdir(label_path):
                    if file_name.endswith('.csv'):
                        file_list.append((os.path.join(label_path, file_name), int(label.split('_')[-1])))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        data = pd.read_csv(file_path, header=None)

        data_parts = []
        if self.use_mic_data and self.use_vacuum:
            mic_vac_data = data.iloc[:, 8:40].values.T
            data_parts.append(mic_vac_data)

        elif self.use_mic_data:
            # mic_data = data.iloc[:, 7:39].values.T
            mic_data = data.iloc[:, 8:39].values.T
            # mic_data = data.iloc[:, 15:18].values.T
            # mic_data[[7, 8, 9], :] *= 2 ## 9~11 채널 2배


            # # 가우시안 가중치 적용
            # freqs = np.arange(31)
            # f0 = 8
            # sigma = 8.0
            # gaussian_weights = np.exp(-0.5 * ((freqs - f0) / sigma)**2)
            # mic_data *= gaussian_weights[:, np.newaxis]


            data_parts.append(mic_data)

        elif self.use_vacuum:
            vacuum = data.iloc[:, 39].values.reshape((1, -1))
            data_parts.append(vacuum)


        if not data_parts:
            raise ValueError("At least one of 'use_vacuum' or 'use_mic_data' must be True.")

        data = np.concatenate(data_parts, axis=0)
        data = torch.from_numpy(data).float()
        return data, label, file_path

# 데이터셋 경로 설정
train_path = gerneral_path+"train"
val_path = gerneral_path+"validation"

# 데이터셋 및 DataLoader 생성
train_dataset = CustomDataset(train_path, use_vacuum=use_vacuum, use_mic_data=use_mic_data)
val_dataset = CustomDataset(val_path, use_vacuum=use_vacuum, use_mic_data=use_mic_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 채널 수 설정
channel_num = train_dataset.channel_num

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 모델 생성 및 초기화
model = LDS1DCNN(input_size=num_input, num_classes=class_num, in_channels=channel_num).to(device)
model.apply(xavier_init)

# 손실 함수 및 최적화 설정
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = StepLR(optimizer, step_size=100, gamma=0.1) 

best_train_accuracy = 0
best_epoch = 0
best_model_wts = None

if __name__ == "__main__":
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        model.train()
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # import pdb;pdb.set_trace()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        print(f'Train Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}, Train Accuracy: {train_accuracy}')

        # 현재 학습률 기록
        train_accuracy_list.append(train_accuracy)

        # 최고 train accuracy 갱신 시 가중치 저장
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            best_epoch = epoch
            best_model_wts = model.state_dict()

        # Validation 루프
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for idx, (val_inputs, val_labels, file_paths) in enumerate(val_loader):
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                _, val_predicted = torch.max(val_outputs.data, 1)

                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_accuracy = val_correct / val_total
        print(f'Validation Epoch {epoch+1}/{num_epochs}, Val Accuracy: {val_accuracy}')
        val_accuracy_list.append(val_accuracy)

    # 최고 정확도의 가중치 저장
    torch.save(best_model_wts, gerneral_path + fig_name +".pth")
    print(f'Best Model saved at epoch {best_epoch+1} with accuracy: {best_train_accuracy:.4f}')



    plt.figure(dpi=500)
    plt.rc('font', size=5)
    plt.plot(range(1, num_epochs+1), train_accuracy_list, marker='o',markersize=3, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), val_accuracy_list, marker='x',markersize=3, label='Validation Accuracy')
    plt.xlabel(f'Epoch / train accuracy : {max(train_accuracy_list)} / val accuracy : {max(val_accuracy_list)}')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0.2,1.2,0.05))
    plt.title(fig_name)
    plt.legend()
    plt.savefig(gerneral_path + fig_name + '.png')
    plt.show()
