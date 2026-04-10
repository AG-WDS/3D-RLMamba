# A Novel Benchmark for Rice Lodging Assessment with X-Band UAV-PolSAR and 3D-RLMamba
This work proposes a novel framework for assessing rice lodging that combines the cutting-edge Unmanned Aerial Vehicle Polarimetric Synthetic Aperture Radar (UAV-PolSAR) with the state-of-the-art (SOTA) 3D Rice Lodging Mamba (3D-RLMamba). This novel framework establishes a
comprehensive benchmark for the intelligent inversion of key lodging parameters, detailing aspects from data acquisition and feature construction to algorithm development and parameter calculation.  
## Note: The dataset will be fully released after our other papers, which are currently under review, are officially published, which will take approximately 6 months.

# ⚠ DATA & CODE Usage
> [!IMPORTANT]
> **Restrictions:** The shared dataset and code are restricted to validation and comparative analysis. Any use of this data for independent publications is prohibited in the absence of additional licensing or permissions.

# 📥 Dataset Download

Google Drive:
 **https://drive.google.com/file/d/1UNJHKd-LxBW74kwrvmZt4IktP9adHYda/view?usp=drive_link**

# 📘 Dataset

This dataset comes from UAV-mounted SAR observations and is designed for rice lodging detection and related research tasks.
It provides training and testing sets along with a 15-layer SAR feature matrix, suitable for both machine learning and deep learning models.

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

# 📜 Citation
    @article
       {Dashuai Wang et al., 2026,
       tilte={A Novel Benchmark for Rice Lodging Assessment with X-Band UAV-PolSAR and 3D-RLMamba},
       author={Dashuai Wang, Minghu Zhao, Zilin Wang, Kunbiao Lu, Changxing Geng*, Xiaoguang Liu*},
       journal={Computers and Electronics in Agriculture},
       doi={https://doi.org/10.1016/j.compag.2026.111730},
       volume={247},
       article number={111730},
       year={2026}
    }

