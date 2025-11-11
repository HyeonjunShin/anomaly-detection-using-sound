import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from umap import UMAP

# -----------------------------------------------------------
# 파일 목록
# -----------------------------------------------------------
file_paths = [
    "data/fft_data/2025-11-11_07-38-32(kim,rubbing,new device).csv",
    "data/fft_data/2025-11-11_07-41-44(kim,crumple,new device).csv",
    "data/fft_data/2025-11-11_07-46-25(shin,rubbing,new device).csv",
    "data/fft_data/2025-11-11_10-58-11(shin,crumple,new device).csv",
    "data/fft_data/2025-11-11_11-04-25(shin,idle, new deivce).csv",
]

# -----------------------------------------------------------
# label 추출
# -----------------------------------------------------------
def extract_label(path):
    if "rubbing" in path:
        return "rubbing"
    elif "crumple" in path:
        return "crumple"
    elif "idle" in path:
        return "idle"
    else:
        return "unknown"

# -----------------------------------------------------------
# 데이터 로드 및 통합
# -----------------------------------------------------------
dfs = []
for path in file_paths:
    df = pd.read_csv(path)
    df["label"] = extract_label(path)
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

# -----------------------------------------------------------
# 주파수 대역 설정 (1~10, 11~20, 21~31)
# -----------------------------------------------------------
bands = {
    "Low (1–10)": (1, 11),
    "Mid (11–20)": (11, 21),
    "High (21–31)": (21, 32),
}

# -----------------------------------------------------------
# UMAP + PCA 기반 시각화 함수
# -----------------------------------------------------------
def visualize_umap_for_band(df, start, end, label, ax):
    mic1 = df.iloc[:, start:end]
    mic2 = df.iloc[:, start+32:end+32]  # mic2_0~31 오프셋 반영
    X = pd.concat([mic1, mic2], axis=1)
    y = df["label"]

    # 표준화
    X_scaled = StandardScaler().fit_transform(X)
    X_pca20 = PCA(n_components=20).fit_transform(X_scaled)

    # 샘플링 (최대 2000개)
    sample_size = 2000
    if len(X_pca20) > sample_size:
        idx = np.random.choice(len(X_pca20), sample_size, replace=False)
        X_sample = X_pca20[idx]
        y_sample = np.array(y)[idx]
    else:
        X_sample = X_pca20
        y_sample = np.array(y)

    # UMAP 적용
    umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
    X_umap = umap.fit_transform(X_sample)

    # Plot
    for lbl in np.unique(y_sample):
        mask = (y_sample == lbl)
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], label=lbl, alpha=0.6)
    ax.set_title(f"{label}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

# -----------------------------------------------------------
# 🎨 주파수 대역별 UMAP subplot 시각화
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (label, (start, end)) in zip(axes, bands.items()):
    visualize_umap_for_band(data, start, end, label, ax)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
fig.suptitle("UMAP Visualization by Frequency Band", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.show()

# -----------------------------------------------------------
# 🤖 RandomForest Baseline (전체 대역)
# -----------------------------------------------------------
mic1_all = data.iloc[:, 1:33]
mic2_all = data.iloc[:, 33:65]
X_all = pd.concat([mic1_all, mic2_all], axis=1)
y_all = data["label"]

X_scaled_all = StandardScaler().fit_transform(X_all)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_all, y_all, test_size=0.25, random_state=42, stratify=y_all
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== RandomForest Classification Report ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_all))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y_all), yticklabels=np.unique(y_all))
plt.title("Confusion Matrix (RandomForest)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
