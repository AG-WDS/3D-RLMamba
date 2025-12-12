# 📘 Dataset

This dataset comes from UAV-mounted SAR observations and is designed for rice lodging detection and related research tasks.
 It provides training and testing sets along with a 15-layer SAR feature matrix, suitable for both machine learning and deep learning models.

## 📥 Dataset Download

Google Drive:
 **https://drive.google.com/file/d/1UNJHKd-LxBW74kwrvmZt4IktP9adHYda/view?usp=drive_link**

## 🌾 15-layer SAR Feature Description (Feature List)

The dataset contains **15 SAR feature channels** representing scattering properties, structural characteristics, and vegetation indices relevant to rice lodging.

| Index | Feature Name                           | Abbreviation | Description                                                  |
| ----- | -------------------------------------- | ------------ | ------------------------------------------------------------ |
| 1     | HH backscatter                         | HH           | Horizontal-Horizontal polarization                           |
| 2     | VV backscatter                         | VV           | Vertical-Vertical polarization                               |
| 3     | HV backscatter                         | HV           | Horizontal-Vertical polarization                             |
| 4     | VH backscatter                         | VH           | Vertical-Horizontal polarization                             |
| 5     | Radar Vegetation Index                 | RVI          | Vegetation structure and density                             |
| 6     | Co-polarization Ratio                  | CPR          | Co-polarization scattering ratio                             |
| 7     | Cross/Co-polarization Ratio            | CCPR         | Cross-to-co-polarization ratio                               |
| 8     | Canopy Structure Index                 | CSI          | Vegetation canopy structure index                            |
| 9     | Radar Forest Degradation Index         | RFDI         | Radar forest degradation index (can reflect lodging structure changes) |
| 10    | Normalized Polarization Index          | NPI          | Normalized polarization index                                |
| 11    | Dual Polarization SAR Vegetation Index | DPSVI        | Dual-polarization vegetation index                           |
| 12    | Modified DPSVI                         | DPSVIm       | Modified dual-polarization vegetation index                  |
| 13    | Pauli Decomposition – R                | PauDecomp-R  | Pauli decomposition red channel                              |
| 14    | Pauli Decomposition – G                | PauDecomp-G  | Pauli decomposition green channel                            |
| 15    | Pauli Decomposition – B                | PauDecomp-B  | Pauli decomposition blue channel                             |

## 📂 Data Structure (Inside `RiceLodging_SAR.mat`)

| Key     | Description           |
| ------- | --------------------- |
| `input` | N × 15 feature matrix |
| `TR`    | Training labels       |
| `TE`    | Testing labels        |

## ⚡ Quick Start

```python
from scipy.io import loadmat
import numpy as np

# 1. Load .mat data
data = loadmat('./data/RiceLodging_SAR.mat')

TR = data['TR']            # Training labels
TE = data['TE']            # Testing labels
input_data = data['input'] # Feature matrix (samples × 15)

print("Input shape:", input_data.shape)
print("Training labels shape:", TR.shape)
print("Testing labels shape:", TE.shape)
```

## 📜 Citation

If you use this dataset in your research, please cite:

```bash
Wang et al., A Novel Benchmark for Rice Lodging Assessment with X-Band UAV-PolSAR and 3D-RLMamba, 2025.
GitHub: https://github.com/AG-WDS/3D-RLMamba
```
