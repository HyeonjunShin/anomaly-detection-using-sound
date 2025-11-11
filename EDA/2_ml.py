import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 파일 목록
file_paths = [
    "data/fft_data/2025-11-11_07-38-32(kim,rubbing,new device).csv",
    "data/fft_data/2025-11-11_07-41-44(kim,crumple,new device).csv",
    "data/fft_data/2025-11-11_07-46-25(shin,rubbing,new device).csv",
    "data/fft_data/2025-11-11_10-58-11(shin,crumple,new device).csv",
    "data/fft_data/2025-11-11_11-04-25(shin,idle, new deivce).csv",
]

# label 자동 추출 (파일명에서 rubbing/crumple/idle 등)
def extract_label(path):
    if "rubbing" in path:
        return "rubbing"
    elif "crumple" in path:
        return "crumple"
    elif "idle" in path:
        return "idle"
    else:
        return "unknown"

# 데이터 로드 및 병합
dfs = []
for path in file_paths:
    df = pd.read_csv(path)
    df["label"] = extract_label(path)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# 🔸 특정 주파수 대역만 선택 (예: 1~10 bin)
mic1_low = data.iloc[:, 1:11]   # mic1_0 ~ mic1_9
mic2_low = data.iloc[:, 33:43]  # mic2_0 ~ mic2_9
X = pd.concat([mic1_low, mic2_low], axis=1)
y = data["label"]

# ---- 📊 EDA: PCA 시각화 ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
for label in np.unique(y):
    mask = (y == label)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.5)
plt.title("PCA Projection (Low Frequency 1~10)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

# ---- 🌈 t-SNE (비선형 시각화) ----
print("Running t-SNE... (takes some time)")
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(8,6))
for label in np.unique(y):
    mask = (y == label)
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=label, alpha=0.6)
plt.title("t-SNE Visualization (Low Frequency 1~10)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend()
plt.show()

# ---- 🤖 RandomForest Baseline ----
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== RandomForest Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix 시각화
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix (RandomForest)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
