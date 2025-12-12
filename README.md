# 📘 Dataset

本数据集来自无人机（UAV）搭载的 SAR 观测，用于水稻倒伏检测等研究任务。
提供训练集、测试集及 15 层 SAR 特征矩阵，适用于机器学习和深度学习模型。

## 📥 Dataset Download

Google Drive：
 **https://drive.google.com/file/d/1UNJHKd-LxBW74kwrvmZt4IktP9adHYda/view?usp=drive_link**



## 🌾 15 层 SAR 特征说明（Feature List）

本数据集包含 **15 个 SAR 特征通道**（用于表征水稻倒伏的散射特性、结构特性与植被指数等）。

| Index | Feature Name                           | Abbreviation | Description                            |
| ----- | -------------------------------------- | ------------ | -------------------------------------- |
| 1     | HH backscatter                         | HH           | Horizontal-Horizontal 极化散射         |
| 2     | VV backscatter                         | VV           | Vertical-Vertical 极化散射             |
| 3     | HV backscatter                         | HV           | Horizontal-Vertical 极化散射           |
| 4     | VH backscatter                         | VH           | Vertical-Horizontal 极化散射           |
| 5     | Radar Vegetation Index                 | RVI          | 植被结构与密度信息                     |
| 6     | Co-polarization Ratio                  | CPR          | 共极化散射比                           |
| 7     | Cross/Co-polarization Ratio            | CCPR         | 交叉/共极化比                          |
| 8     | Canopy Structure Index                 | CSI          | 植被冠层结构指数                       |
| 9     | Radar Forest Degradation Index         | RFDI         | 雷达森林退化指数（可用于倒伏结构变化） |
| 10    | Normalized Polarization Index          | NPI          | 归一化极化指数                         |
| 11    | Dual Polarization SAR Vegetation Index | DPSVI        | 双极化植被指数                         |
| 12    | Modified DPSVI                         | DPSVIm       | 改进双极化植被指数                     |
| 13    | Paul Decomposition – R                 | PauDecomp-R  | Pauli 分解红色通道                     |
| 14    | Paul Decomposition – G                 | PauDecomp-G  | Pauli 分解绿色通道                     |
| 15    | Paul Decomposition – B                 | PauDecomp-B  | Pauli 分解蓝色通道                     |

## 📂 数据结构（Inside `RiceLodging_SAR.mat`）

| Key     | Description                   |
| ------- | ----------------------------- |
| `input` | N × 15 的特征矩阵             |
| `TR`    | 训练集标签（Training labels） |
| `TE`    | 测试集标签（Testing labels）  |

## ⚡ 快速使用（Quick Start）

```python
from scipy.io import loadmat
import numpy as np

# 1. 加载 .mat 数据
data = loadmat('./data/RiceLodging_SAR.mat')

TR = data['TR']            # 训练集标签
TE = data['TE']            # 测试集标签
input_data = data['input'] # 特征矩阵 (samples × 15)

print("Input shape:", input_data.shape)
print("Training labels shape:", TR.shape)
print("Testing labels shape:", TE.shape)
```

## 📜 Citation

如果你在研究中使用本数据集，请引用：

```
Wang et al., A Novel Benchmark for Rice Lodging Assessment with X-Band UAV-PolSAR and 3D-RLMamba, 2025.
GitHub: https://github.com/AG-WDS/3D-RLMamba
```
