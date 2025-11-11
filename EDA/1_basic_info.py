import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# --------------------------------------------------------------------
# 사용자 설정
file_paths = [
    "data/fft_data/2025-11-11_07-38-32(kim,rubbing,new device).csv",
    "data/fft_data/2025-11-11_07-41-44(kim,crumple,new device).csv",
    "data/fft_data/2025-11-11_07-46-25(shin,rubbing,new device).csv",
    "data/fft_data/2025-11-11_10-58-11(shin,crumple,new device).csv",
    "data/fft_data/2025-11-11_11-04-25(shin,idle, new deivce).csv",
]

freq_range = (20, 30)   # ✅ 분석할 주파수 bin 구간 설정 (예: 1~10)
# --------------------------------------------------------------------

def extract_label(path):
    name = os.path.basename(path)
    for label in ["rubbing", "crumple", "idle"]:
        if label in name:
            return label
    return "unknown"

# 특정 주파수 대역만 추출하는 함수
def get_freq_band(df, start, end):
    mic1_cols = df.filter(like='mic1_').columns[start:end]
    mic2_cols = df.filter(like='mic2_').columns[start:end]
    mic1 = df[mic1_cols]
    mic2 = df[mic2_cols]
    return mic1, mic2

# --------------------------------------------------------
# 데이터 로드 및 라벨링
dfs = []
for path in file_paths:
    df = pd.read_csv(path)
    df["label"] = extract_label(path)
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

# 주파수 대역 적용
mic1_band, mic2_band = get_freq_band(data, *freq_range)
band_df = pd.concat([mic1_band, mic2_band, data["label"]], axis=1)
print(f"선택된 주파수 구간: mic1_{freq_range[0]} ~ mic1_{freq_range[1]-1}")

# PCA 계산
X = band_df.filter(regex='mic[12]_').values
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
band_df["pca1"], band_df["pca2"] = X_pca[:,0], X_pca[:,1]

# --------------------------------------------------------
# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14,10))
fig.suptitle(f"FFT Audio EDA (Freq {freq_range[0]}~{freq_range[1]-1})", fontsize=16, weight="bold")

# (1) 평균 스펙트럼 (Mic1)
ax = axes[0,0]
for label, grp in band_df.groupby("label"):
    mic1_mean = grp.filter(like='mic1_').mean().values
    ax.plot(mic1_mean, label=label)
ax.set_title("Average Spectrum (Mic1)")
ax.set_xlabel("Frequency Bin")
ax.set_ylabel("Amplitude")
ax.legend()

# (2) Mic1 vs Mic2 평균 비교
ax = axes[0,1]
for label, grp in band_df.groupby("label"):
    mic1_mean = grp.filter(like='mic1_').mean().values
    mic2_mean = grp.filter(like='mic2_').mean().values
    ax.plot(mic1_mean, linestyle='-', label=f"{label} Mic1")
    ax.plot(mic2_mean, linestyle='--', label=f"{label} Mic2")
ax.set_title("Mic1 vs Mic2 Comparison")
ax.legend(fontsize=8)

# (3) PCA 시각화
ax = axes[1,0]
sns.scatterplot(data=band_df, x="pca1", y="pca2", hue="label", s=10, alpha=0.7, ax=ax)
ax.set_title("PCA projection (selected freq band)")
ax.legend(fontsize=8)

# (4) 샘플 스펙트로그램
ax = axes[1,1]
sample_path = file_paths[0]
sample_label = extract_label(sample_path)
df_sample = pd.read_csv(sample_path)
mic1_sample, _ = get_freq_band(df_sample, *freq_range)
mic1_db = 20 * np.log10(mic1_sample.clip(lower=1))
sns.heatmap(mic1_db.iloc[:200,:], cmap="magma", ax=ax, cbar=False)
ax.set_title(f"{sample_label} Spectrogram (Mic1, bins {freq_range[0]}~{freq_range[1]-1})")
ax.set_xlabel("Frequency Bin")
ax.set_ylabel("Time Frame")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
