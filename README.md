<div align="center">

# 🔍 Multi-Scale PCB Defect Detection: From Custom Architectures to MC-VAM Enhanced YOLOv11

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![Hardware](https://img.shields.io/badge/Trained_on-NVIDIA_Tesla_T4-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![Academic Project](https://img.shields.io/badge/UNIMIB-Foundations_of_Deep_Learning-003366?style=for-the-badge)](https://www.unimib.it/)

**An end-to-end deep learning study on automated Printed Circuit Board (PCB) surface defect detection, evaluating custom CNNs from scratch, ResNet-34 multi-scale feature extractors, YOLOv11s baseline, and an attention-augmented YOLOv11 with Multi-Cognitive Visual Attention (MC-VAM).**

---

### 👥 Authors & Academic Context

- **Ahmed Aly Maged Mohamed Mohamed Sayed** (Matricola: `946984`)
- **Askar Amro Emad Hassan** (Matricola: `946807`)

**Institution:** University of Milano-Bicocca (UNIMIB)  
**Course:** Foundations of Deep Learning (FDL) — Academic Year 2025/2026

---

</div>

## 📑 Table of Contents

1. [Executive Summary](#-executive-summary)
2. [The Problem: Micro-Defects in High-Resolution PCBs](#-the-problem-micro-defects-in-high-resolution-pcbs)
3. [Dataset & Exploratory Data Analysis (EDA)](#-dataset--exploratory-data-analysis-eda)
4. [Iterative Methodological Progression](#-iterative-methodological-progression)
   - [Notebook 1: Exploratory Data Analysis (EDA)](#notebook-1-exploratory-data-analysis-eda)
   - [Notebook 2: From-Scratch Grid CNN Detector (Failed Trial Analysis)](#notebook-2-from-scratch-grid-cnn-detector-failed-trial-analysis)
   - [Notebook 3: ResNet-34 Backbone with Multi-Scale Feature Fusion](#notebook-3-resnet-34-backbone-with-multi-scale-feature-fusion)
   - [Notebook 4: YOLOv11s SOTA Baseline](#notebook-4-yolov11s-sota-baseline)
   - [Notebook 5: MC-VAM Enhanced YOLOv11s + Hyperparameter Grid Search](#notebook-5-mc-vam-enhanced-yolov11s--hyperparameter-grid-search)
5. [MC-VAM: Multi-Cognitive Visual Attention Module](#-mc-vam-multi-cognitive-visual-attention-module)
6. [Experimental Results & Benchmarks](#-experimental-results--benchmarks)
   - [Model Comparison Table](#model-comparison-table)
   - [Hyperparameter Grid Search (MC-VAM)](#hyperparameter-grid-search-mc-vam)
   - [Key Insights & Findings](#key-insights--findings)
7. [Repository Structure](#-repository-structure)
8. [Trained Models & Checkpoints](#-trained-models--checkpoints)
9. [Installation & Setup](#-installation--setup)
10. [How to Run & Reproduce](#-how-to-run--reproduce)
11. [Presentation Slides](#-presentation-slides)
12. [References & Citations](#-references--citations)

---

## 📌 Executive Summary

Manual inspection of Printed Circuit Boards (PCBs) is prone to human fatigue, time-consuming, and expensive. In automated industrial inspection, the central challenge stems from **extreme scale disparity**: PCB defects (e.g., micro-cuts, hairline shorts, tiny pinhole omissions) typically measure under **$50 \times 50$ to $80 \times 80$ pixels** on ultra-high-resolution board scans exceeding **$3000 \times 2500$ pixels** (< 0.1% of the total area).

This project investigates the limits and strengths of deep learning architectures on the PKU PCB dataset through a systematic 5-stage evolutionary roadmap:

1. **From-Scratch CNN (`PCBDetectorV3`, ~1.65M params):** Built a custom grid-based detector. Revealed fundamental representation bottlenecks when training from scratch on limited samples of micro-scale targets.
2. **Transfer Learning via ResNet-34 (~22.89M params):** Extracted multi-scale representations ($C_3, C_4$), fused semantic and spatial contexts, achieving **78.14% Precision, 75.17% Recall, and 95.71% image-level classification accuracy**.
3. **YOLOv11s Baseline:** Leveraged modern SOTA real-time detection featuring SPPF, C2PSA, and C3k2 blocks, scoring **96.9% Precision and 96.7% mAP@50**.
4. **Architectural Enhancement (MC-VAM + YOLOv11s):** Adapted a multi-scale depthwise convolution visual attention block proposed by _Jin et al. (Scientific Reports, 2026)_, pushing performance to **97.1% Precision and 97.2% mAP@50**.
5. **Hyperparameter Grid Search:** Evaluated 11 configurations across image resolutions (640 vs. 800), learning rates, and batch sizes, demonstrating that higher spatial resolution directly mitigates small-object degradation.

---

## 🔬 The Problem: Micro-Defects in High-Resolution PCBs

The inspection process involves detecting, categorizing, and localizing six critical defect classes:

| Defect Class        | Visual Characteristics & Industrial Impact                                                                |
| :------------------ | :-------------------------------------------------------------------------------------------------------- |
| **Missing Hole**    | Un-drilled or blocked via/through-hole; prevents component insertion and electrical interconnects.        |
| **Mouse Bite**      | Small edge breakouts or notches in copper traces or laminate, threatening mechanical/thermal integrity.   |
| **Open Circuit**    | Break in a conductive track, resulting in complete circuit discontinuity and operational failure.         |
| **Short**           | Unintended conductive bridge bridging two adjacent tracks, causing electrical short-circuits.             |
| **Spur**            | Sharp unwanted copper protrusion projecting from a trace, causing impedance variations or flashover risk. |
| **Spurious Copper** | Isolated speck of copper residue on substrate, posing leakage or unintended capacitive coupling hazards.  |

---

## 📊 Dataset & Exploratory Data Analysis (EDA)

The project utilizes the benchmark synthetic **PKU PCB Dataset** ([Tang et al., 2019](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/joe.2019.1183)):

- **Total Images:** 693 high-resolution RGB images across 10 template layouts.
- **Image Resolutions:** Ranging from $2240 \times 1586$ up to $3056 \times 2530$ pixels.
- **Total Defect Annotations:** 2,953 bounding box instances (average ~4.26 defects per board).
- **Dataset Splits:** Stratified into **Train: 485 images (70%)**, **Validation: 138 images (20%)**, and **Test: 70 images (10%)**.

### Class Distribution & Scale Disparity

The dataset maintains high class balance across all 6 defect categories, but exhibits extreme relative smallness:

| Defect Class        | Total Instances | Average Width (px) | Average Height (px) | Relative Area vs Board |
| :------------------ | :-------------: | :----------------: | :-----------------: | :--------------------: |
| **Missing Hole**    |       497       |       66.76        |        68.10        |         ~0.08%         |
| **Mouse Bite**      |       492       |       63.07        |        61.43        |         ~0.06%         |
| **Open Circuit**    |       482       |     **52.36**      |      **49.79**      | **~0.04% (Smallest)**  |
| **Short**           |       491       |       86.74        |        78.75        |         ~0.11%         |
| **Spur**            |       488       |       75.41        |        69.36        |         ~0.09%         |
| **Spurious Copper** |       503       |       80.21        |        77.48        |         ~0.10%         |
| **Total / Overall** |    **2,953**    |     **70.76**      |      **67.48**      |      **< 0.10%**       |

```
Standard Board Size: ~2800 x 2000 pixels
+-------------------------------------------------------------+
|                                                             |
|   PCB Substrate                                             |
|                                                             |
|          [.] <- Defect (avg ~60x60 px)                      |
|                                                             |
|                                                             |
+-------------------------------------------------------------+
Downsampling to 640x640 collapses a 50x50 defect down to ~11x11 pixels!
```

---

## 🛠 Iterative Methodological Progression

### Notebook 1: Exploratory Data Analysis (EDA)

📁 [`Codes/1-eda-deeplearningprojectunimib.ipynb`](file:///d:/Masters/Year%201/Semester%202/Deep%20Learning/Project/Codes/1-eda-deeplearningprojectunimib.ipynb)

- Extracted and parsed Pascal VOC XML annotations.
- Quantified bounding box aspect ratios, spatial centroid distributions, and class frequencies.
- Analyzed image dimensions and investigated the effect of rotation artifacts (`rotate.py` white border padding).
- Laid the empirical groundwork establishing that standard downsampling degrades tiny defects.

---

### Notebook 2: From-Scratch Grid CNN Detector (Failed Trial Analysis)

📁 [`Codes/2-fromscratch-deeplearningprojectunimib.ipynb`](file:///d:/Masters/Year%201/Semester%202/Deep%20Learning/Project/Codes/2-fromscratch-deeplearningprojectunimib.ipynb)

- **Architecture:** `PCBDetectorV3` (~1.65M parameters).
  - 3 downsampling stages: $800 \times 800 \to 400 \times 400 \to 200 \times 200 \to 100 \times 100$ feature map.
  - Output tensor: $100 \times 100 \times 11$ (1 objectness logit + 6 class logits + 4 box coordinate offsets per grid cell).
- **Multi-Task Loss:**
  $$\mathcal{L} = \mathcal{L}_{obj} + \mathcal{L}_{cls} + 5 \cdot \mathcal{L}_{box}$$
  - $\mathcal{L}_{obj}$: `BCEWithLogitsLoss` with positive weighting (`pos_weight = 20.0`).
  - $\mathcal{L}_{cls}$: Multi-class `CrossEntropyLoss` on positive cells.
  - $\mathcal{L}_{box}$: `SmoothL1Loss` on normalized center coordinates $(t_x, t_y)$ and dimensions $(t_w, t_h)$.
- **Failure Mode & Autopsy:**
  - With only 485 training images and extreme foreground-to-background imbalance (only ~4 positive cells out of 10,000 cells per image), the network struggled to converge on reliable spatial representations.
  - The model produced a high rate of false positive alarms (averaging ~8.6 predictions per image), confirming that training a deep detector entirely from scratch without pretrained weights requires significantly larger datasets and specialized anchor pyramid mechanisms.

---

### Notebook 3: ResNet-34 Backbone with Multi-Scale Feature Fusion

📁 [`Codes/3-resnet_backbone-deeplearningprojectunimib.ipynb`](file:///d:/Masters/Year%201/Semester%202/Deep%20Learning/Project/Codes/3-resnet_backbone-deeplearningprojectunimib.ipynb)

- **Architecture:** Hybrid Transfer Learning Detector (~22.89M parameters).
  - **Backbone:** ImageNet-pretrained `ResNet-34` extracting multi-scale feature hierarchies:
    - Stage $C_3$: 256 channels, stride 16 (retaining fine spatial resolution).
    - Stage $C_4$: 512 channels, stride 32 (rich semantic context).
  - **Feature Fusion Module:** Projects $C_4$ to 256 channels via $1 \times 1$ conv, upsamples by $2\times$ via bilinear interpolation, adds to $C_3$, and smooths via $3 \times 3$ conv + BatchNorm + ReLU.
  - **Detection Head:** Shared $256 \to 128$ intermediate representation branching into:
    - Objectness Head ($1 \times 1$ conv $\to 1$ channel).
    - Classification Head ($1 \times 1$ conv $\to 6$ channels).
    - Box Regression Head ($1 \times 1$ conv $\to 4$ channels).
- **Loss Configuration:** Balanced objective with `pos_weight = 50.0`.
- **Results on Hold-Out Test Set (70 images):**
  - **Precision:** `78.14%`
  - **Recall:** `75.17%`
  - **F1-Score:** `76.63%`
  - **Mean IoU:** `0.4216`
  - **Image-Level Defect Classification Accuracy:** `95.71%` (95.71% of boards had their dominant defect correctly recognized).

---

### Notebook 4: YOLOv11s SOTA Baseline

📁 [`Codes/4-yolo11baseline-deeplearningprojectunimib.ipynb`](file:///d:/Masters/Year%201/Semester%202/Deep%20Learning/Project/Codes/4-yolo11baseline-deeplearningprojectunimib.ipynb)

- Formatted annotations into normalized YOLO coordinate format `(class_idx, x_center, y_center, width, height)`.
- Fine-tuned pretrained `yolo11s.pt` with data augmentations: horizontal/vertical flip (0.5), affine rotation ($\pm 20^\circ$), translation (0.1), scale jitter (0.2), and Mosaic augmentation (1.0).
- **Test Performance (Input size 800, batch 8):**
  - **Precision:** `0.969` (96.9%)
  - **Recall:** `0.947` (94.7%)
  - **mAP@50:** `0.967` (96.7%)
  - **mAP@50-95:** `0.529` (52.9%)

---

### Notebook 5: MC-VAM Enhanced YOLOv11s + Hyperparameter Grid Search

📁 [`Codes/5-yolo11+mcvam-deeplearningprojectunimib.ipynb`](file:///d:/Masters/Year%201/Semester%202/Deep%20Learning/Project/Codes/5-yolo11+mcvam-deeplearningprojectunimib.ipynb)

- Extended the core `ultralytics` library by injecting the **Multi-Cognitive Visual Attention Module (MC-VAM)** into `ultralytics/nn/modules/block.py`.
- Configured custom model definition `yolo11s_mcvam.yaml` by inserting MC-VAM downstream of layer 10 (C2PSA), enriching deep feature extraction before multi-scale neck aggregation.
- Conducted systematic grid search exploring **Input Size** ($640$ vs. $800$), **Learning Rate** ($0.01, 0.005, 0.001$), and **Batch Size** ($8$ vs. $16$).
- **Best Test Performance:**
  - **Precision:** `0.971` (97.1%) — _improvement over baseline_
  - **Recall:** `0.946` (94.6%)
  - **mAP@50:** `0.972` (97.2%) — _highest overall detection accuracy_
  - **mAP@50-95:** `0.527` (52.7%)

---

## 🧠 MC-VAM: Multi-Cognitive Visual Attention Module

> [!IMPORTANT]
>
> ### 📜 Academic Attribution & Acknowledgement
>
> The **Multi-Cognitive Visual Attention Module (MC-VAM)** is **not our original architectural invention**. It was designed and proposed by **Lingxiao Jin, Yuqin Feng, Hao Yang, and Shuxian Liu** in their research paper:  
> **["A multi-cognitive PCB defect detection model integrating Mamba"](https://doi.org/10.1038/s41598-026-49734-2)**, published in _Scientific Reports_ (Nature Portfolio), 2026.  
> **Our contribution in this project** lies in studying this novel attention block, adapting and reimplementing its core multi-scale receptive field mechanism into the modern **Ultralytics YOLOv11s** architecture as a modular plug-and-play block, and conducting an extensive hyperparameter grid search on the PKU PCB dataset.

### 📖 Motivation & Mathematical Formulation from the Paper

In deep convolutional and hybrid object detectors, deep layers feature high semantic abstraction but often suffer from severe loss of shallow spatial details. For PCB inspection—where defects like micro-breaks (_Open circuit_) or minute copper specks (_Spurious copper_) span only dozens of pixels—standard downsampling leads to target attenuation.

To resolve this, Jin et al. introduced **MC-VAM** to strengthen multi-scale perception and preserve shallow features through parallel multi-receptive depthwise convolutions and residual paths:

1. **Feature Normalization & Scaling:**
   $$\mathbf{X}_{norm} = s_1 \odot \text{LN}(\mathbf{X}_{in}) + s_2 \odot \mathbf{X}_{in}$$
   where $\text{LN}(\cdot)$ is layer/channel normalization and $s_1, s_2$ are learnable scaling parameters.

2. **Down-Projection:**
   $$\mathbf{X}_{down} = \mathbf{W}_{down} * \mathbf{X}_{norm}$$
   A $1 \times 1$ convolution reduces channels to constrain computational overhead.

3. **Multi-Scale Depthwise Convolutions:**
   Parallel depthwise separable convolutions with kernel sizes $3 \times 3$, $5 \times 5$, and $7 \times 7$ capture fine, medium, and broad spatial contexts simultaneously without cross-channel parameter bloat:
   $$\mathbf{X}_{dw} = \mathbf{X}_{down} + \frac{1}{3} \sum_{k \in \{3, 5, 7\}} \left( \mathbf{W}_{dw}^{k \times k} * \mathbf{X}_{down} \right)$$

4. **Pointwise Cross-Channel Interaction:**
   $$\mathbf{X}_{pw} = \mathbf{X}_{dw} + \mathbf{W}_{pw} * \mathbf{X}_{dw}$$

5. **Activation & Deep Residual Connection:**
   $$\mathbf{X}_{out} = \mathbf{X}_{in} + \text{Up}\left(\text{Activation}(\mathbf{X}_{pw})\right)$$
   The deep residual connection directly transfers shallow, high-frequency spatial cues back to the output, preserving tiny defect contours.

```
                         Input Tensor [B, C, H, W]
                                    |
                           +--------+--------+ (Deep Residual Shortcut)
                           |                 |
                           v                 |
                    Norm / GroupNorm         |
                           |                 |
                           v                 |
                   1x1 Conv (Down-Proj)      |
                           |                 |
           +---------------+---------------+ |
           |               |               | |
           v               v               v |
       3x3 DW-Conv     5x5 DW-Conv     7x7 DW-Conv
      (Local Detail) (Medium Context) (Global Context)
           |               |               | |
           +---------------+---------------+ |
                           |                 |
                           v                 |
                    Average Fusion           |
                     (b1 + b2 + b3)/3        |
                           |                 |
                           v                 |
                    1x1 Conv (Fuse)          |
                           |                 |
                           v                 |
                   Non-linear Activation     |
                           |                 |
                           v                 |
                         (+) <---------------+
                           |
                           v
                     Output Tensor
```

### 💻 Our Implementation & Integration into YOLOv11s:

In [`Codes/5-yolo11+mcvam-deeplearningprojectunimib.ipynb`](file:///d:/Masters/Year%201/Semester%202/Deep%20Learning/Project/Codes/5-yolo11+mcvam-deeplearningprojectunimib.ipynb), we adapted MC-VAM into PyTorch and registered it within the Ultralytics module registry:

```python
import torch
import torch.nn as nn

class MCVAM(nn.Module):
    """
    Multi-Cognitive Visual Attention Module (MC-VAM).
    Originally designed by Jin et al. (Scientific Reports, 2026).
    Adapted here for modular integration into Ultralytics YOLOv11s.
    """
    def __init__(self, c1, c2=None, *args, **kwargs):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(32, c1), num_channels=c1)
        self.reduce = nn.Conv2d(c1, c1, kernel_size=1, bias=False)
        self.dw3 = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False)
        self.dw5 = nn.Conv2d(c1, c1, kernel_size=5, padding=2, groups=c1, bias=False)
        self.dw7 = nn.Conv2d(c1, c1, kernel_size=7, padding=3, groups=c1, bias=False)
        self.fuse = nn.Conv2d(c1, c1, kernel_size=1, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x_norm = self.reduce(self.norm(x))
        b1 = self.dw3(x_norm)
        b2 = self.dw5(x_norm)
        b3 = self.dw7(x_norm)
        fused = (b1 + b2 + b3) / 3.0
        out = self.act(self.fuse(fused))
        return out + residual
```

In `yolo11s_mcvam.yaml`, MC-VAM was inserted right after layer 10 (`C2PSA`), enriching deep feature representation before multiscale neck feature fusion.

---

## 📈 Experimental Results & Benchmarks

### Model Comparison Table

| Model Architecture                 | Pretraining | Parameters |     Input Res      | Precision | Recall  |  F1 / mAP@50   | mAP@50-95 | Status / Notes                                             |
| :--------------------------------- | :---------: | :--------: | :----------------: | :-------: | :-----: | :------------: | :-------: | :--------------------------------------------------------- |
| **From Scratch (`PCBDetectorV3`)** |    None     |   1.65M    |  $800 \times 800$  |    Low    |   Low   |      N/A       |    N/A    | _Failed trial_: Severe false positives due to tiny targets |
| **ResNet-34 + Custom Head**        |  ImageNet   |   22.89M   | $1024 \times 1024$ |  `0.781`  | `0.752` | `0.766` _(F1)_ |    N/A    | **95.71%** Image classification accuracy                   |
| **YOLOv11s Baseline**              |    COCO     |   ~9.4M    |  $800 \times 800$  |  `0.969`  | `0.947` |    `0.967`     | **0.529** | High baseline capability                                   |
| **Enhanced YOLOv11s + MC-VAM**     |    COCO     |   ~9.5M    |  $800 \times 800$  | **0.971** | `0.946` |   **0.972**    |  `0.527`  | **Best detection accuracy & precision**                    |

---

### Hyperparameter Grid Search (MC-VAM)

Exhaustive evaluation of YOLOv11s + MC-VAM across image size, initial learning rate, and batch size (from [`Models/5-Grid_Search_Results(yolo11+mcvam).xlsx`](<file:///d:/Masters/Year%201/Semester%202/Deep%20Learning/Project/Models/5-Grid_Search_Results(yolo11+mcvam).xlsx>)):

| Run Name                       | Image Size | Initial LR | Batch Size | Stopped Epoch | Best Epoch |   mAP@50   | mAP@50-95  | Precision  |   Recall   |
| :----------------------------- | :--------: | :--------: | :--------: | :-----------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| **`img800_lr0.01_bs8` (Best)** |  **800**   | **0.010**  |   **8**    |      154      |    123     | **0.9721** | **0.5270** | **0.9708** |   0.9465   |
| `img800_lr0.005_bs8`           |    800     |   0.005    |     8      |      154      |    123     |   0.9721   |   0.5270   |   0.9708   |   0.9465   |
| `img800_lr0.001_bs8`           |    800     |   0.001    |     8      |      154      |    123     |   0.9721   |   0.5270   |   0.9708   |   0.9465   |
| `img800_lr0.005_bs16`          |    800     |   0.005    |     16     |      197      |    167     |   0.9658   |   0.5245   |   0.9661   | **0.9494** |
| `img800_lr0.001_bs16`          |    800     |   0.001    |     16     |      197      |    167     |   0.9658   |   0.5245   |   0.9661   | **0.9494** |
| `img640_lr0.001_bs16`          |    640     |   0.001    |     16     |      147      |    116     |   0.9577   |   0.5049   |   0.9650   |   0.9216   |
| `img640_lr0.01_bs16`           |    640     |   0.010    |     16     |      147      |    116     |   0.9577   |   0.5049   |   0.9650   |   0.9216   |
| `img640_lr0.005_bs16`          |    640     |   0.005    |     16     |      147      |    116     |   0.9577   |   0.5049   |   0.9650   |   0.9216   |
| `img640_lr0.005_bs8`           |    640     |   0.005    |     8      |      166      |    165     |   0.9446   |   0.5016   |   0.9448   |   0.9116   |
| `img640_lr0.01_bs8`            |    640     |   0.010    |     8      |      166      |    165     |   0.9446   |   0.5016   |   0.9448   |   0.9116   |
| `img640_lr0.001_bs8`           |    640     |   0.001    |     8      |      145      |    134     |   0.9439   |   0.5052   |   0.9468   |   0.9159   |

---

### 🔍 Key Insights & Findings

1. **Resolution is King for Micro-Defects:** Increasing input resolution from 640 to 800 yielded an immediate **+2.8% boost in mAP@50** (from ~0.944 to 0.972). Tiny defects like _Open Circuit_ (~50px) require higher pixel densities to avoid total feature attenuation.
2. **MC-VAM Attention Efficacy:** Adding MC-VAM improved both precision (0.969 $\to$ 0.971) and mAP@50 (0.967 $\to$ 0.972) by giving the network simultaneous $3 \times 3, 5 \times 5$, and $7 \times 7$ receptive fields, making it easier to capture fine-grained copper anomalies.
3. **Small Batch Advantage:** On Tesla T4 GPU, batch size 8 achieved slightly sharper gradient updates and better mAP50 (0.972) compared to batch size 16 (0.966).
4. **Limits of Training from Scratch:** Small industrial datasets cannot supply the feature variety necessary to train deep object detectors from scratch without strong pre-trained priors.

---

## 📂 Repository Structure

```
.
├── Codes/
│   ├── 1-eda-deeplearningprojectunimib.ipynb               # Dataset exploration, statistics & distributions
│   ├── 2-fromscratch-deeplearningprojectunimib.ipynb        # From-scratch CNN detector trial (1.65M params)
│   ├── 3-resnet_backbone-deeplearningprojectunimib.ipynb    # ResNet-34 + multi-scale fusion head (22.89M params)
│   ├── 4-yolo11baseline-deeplearningprojectunimib.ipynb     # Pretrained YOLOv11s baseline detector
│   └── 5-yolo11+mcvam-deeplearningprojectunimib.ipynb       # MC-VAM integration & grid search training
│
├── Models/
│   ├── 3-resnet_backbone_best_pcb_detector.pth             # Best ResNet-34 checkpoint (~91.7 MB)
│   ├── 4-yolo11baselinemodel.pt                            # YOLOv11s baseline weights (~19.2 MB)
│   ├── 5-yolo_mcvam_best.pt                                # Best YOLOv11s + MC-VAM weights (~28.9 MB)
│   ├── 5-Grid_Search_Results(yolo11+mcvam).xlsx            # Complete grid search benchmark spreadsheet
│   └── 5-yolo_mcvam_logs.txt                               # Detailed Kaggle training logs for MC-VAM
│
├── PCB-DATASET-master/                                     # PKU PCB Dataset (Images & XML Annotations)
│   └── PCB-DATASET-master/
│       ├── Annotations/                                    # Pascal VOC XML annotations per defect class
│       ├── images/                                         # High-resolution PCB images
│       ├── PCB_USED/                                       # Cropped templates
│       ├── rotation/                                       # Rotated samples
│       └── rotate.py                                       # Data rotation utility
│
├── PCB-Defect-Detection-AHMED-ASKAR.pdf                    # Project Presentation (PDF)
├── PCB-Defect-Detection-AHMED-ASKAR.pptx                   # Project Presentation (PowerPoint)
├── s41598-026-49734-2_reference.pdf                        # Reference Research Paper: Jin et al. (Sci Rep, 2026)
├── [FDL] Guide to the exam 2025_2026.pdf                   # UNIMIB Course Exam Guide
├── [FDL] Project topics 2025_2026.pdf                      # UNIMIB Approved Topics & Datasets
├── Readme.txt                                              # Original project notes & weights cloud link
└── README.md                                               # Project documentation
```

---

## 💾 Trained Models & Checkpoints

Due to GitHub's file size limits for large binary artifacts, full model checkpoints are accessible both locally in the `Models/` directory and on Google Drive:

- 🔗 **Google Drive Checkpoints Folder:**  
  [Google Drive - Trained Models](https://drive.google.com/drive/folders/16x9bzZVZKJgQLnGWD_RtufEgPlORwXRj?usp=sharing)

| Model Checkpoint                          | Description                                 | File Size | Location          |
| :---------------------------------------- | :------------------------------------------ | :-------: | :---------------- |
| `3-resnet_backbone_best_pcb_detector.pth` | Best weights for ResNet-34 + Feature Fusion |  91.7 MB  | `Models/` & Drive |
| `4-yolo11baselinemodel.pt`                | Best weights for YOLOv11s Baseline          |  19.2 MB  | `Models/` & Drive |
| `5-yolo_mcvam_best.pt`                    | Best weights for YOLOv11s + MC-VAM          |  28.9 MB  | `Models/` & Drive |

---

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and Activate a Virtual Environment

```bash
# Using conda
conda create -n pcb-defect python=3.10 -y
conda activate pcb-defect

# Or using venv
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python matplotlib seaborn pandas openpyxl tqdm
```

---

## 🚀 How to Run & Reproduce

All experiments were developed and verified on **Kaggle Notebooks (Tesla T4 GPU, 16GB VRAM)**.

### Running the Notebooks

To rerun or explore any stage, launch Jupyter Notebook or Jupyter Lab:

```bash
jupyter lab
```

1. **EDA:** Run `Codes/1-eda-deeplearningprojectunimib.ipynb` to inspect distributions, box aspect ratios, and dataset anomalies.
2. **From-Scratch Trial:** Run `Codes/2-fromscratch-deeplearningprojectunimib.ipynb` to see the custom grid detector implementation and inspect its loss curves.
3. **ResNet-34 Model:** Run `Codes/3-resnet_backbone-deeplearningprojectunimib.ipynb` to train the ResNet-34 feature fusion model and evaluate IoU/Precision/Recall on the 70 test images.
4. **YOLOv11 Baseline:** Run `Codes/4-yolo11baseline-deeplearningprojectunimib.ipynb` to convert XML annotations to YOLO format and run the baseline training.
5. **MC-VAM YOLOv11:** Run `Codes/5-yolo11+mcvam-deeplearningprojectunimib.ipynb` to inject MC-VAM into Ultralytics and run hyperparameter validation.

> **Note on paths:** If running locally, ensure you update the dataset path variable (`DATASET_PATH` / `yolo_root`) in each notebook to point to your local `PCB-DATASET-master/` path.

---

## 📽 Presentation Slides

The complete slide deck defending this project is available in the root repository:

- 📄 **PDF Version:** [`PCB-Defect-Detection-AHMED-ASKAR.pdf`](file:///d:/Masters/Year%201/Semester%202/Deep%20Learning/Project/PCB-Defect-Detection-AHMED-ASKAR.pdf)
- 📊 **PowerPoint Version:** [`PCB-Defect-Detection-AHMED-ASKAR.pptx`](file:///d:/Masters/Year%201/Semester%202/Deep%20Learning/Project/PCB-Defect-Detection-AHMED-ASKAR.pptx)

The presentation covers the industrial problem, EDA findings on small defect scales, failure analysis of the from-scratch model, transfer learning with ResNet-34, and the final MC-VAM attention enhancements.

---

## 📚 References & Citations

1. **MC-VAM Attention & PCB-MMF Architecture (Paper acknowledged):**  
   Jin, L., Feng, Y., Yang, H., & Liu, S. (2026). _A multi-cognitive PCB defect detection model integrating Mamba._ **Scientific Reports**, Nature Portfolio.  
   📄 **DOI:** [https://doi.org/10.1038/s41598-026-49734-2](https://doi.org/10.1038/s41598-026-49734-2)  
   _Note: MC-VAM was originally developed and published by Jin et al. in this paper. Our repository adapts their architectural design into YOLOv11s for experimental benchmark purposes._

   ```bibtex
   @article{jin2026multicognitive,
     title={A multi-cognitive PCB defect detection model integrating Mamba},
     author={Jin, Lingxiao and Feng, Yuqin and Yang, Hao and Liu, Shuxian},
     journal={Scientific Reports},
     year={2026},
     publisher={Nature Publishing Group},
     doi={10.1038/s41598-026-49734-2},
     url={https://doi.org/10.1038/s41598-026-49734-2}
   }
   ```

2. **Benchmark PCB Dataset:**  
   Tang, S., He, F., Huang, N., & Li, J. (2019). _A PCB dataset for defect detection via advanced deep learning methods._ The Journal of Engineering, 2019(19), 7082–7086.  
   📄 **DOI:** [10.1049/joe.2019.1183](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/joe.2019.1183) | [arXiv:1901.08204](https://arxiv.org/abs/1901.08204)

3. **YOLOv11 Framework:**  
   Ultralytics Team (2024). _YOLOv11: State-of-the-Art Real-Time Object Detection._  
   💻 [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

4. **ResNet Architecture:**  
   He, K., Zhang, X., Ren, S., & Sun, J. (2016). _Deep Residual Learning for Image Recognition._ IEEE/CVF CVPR 2016.

---

<div align="center">
Ahmed Aly Maged & Askar Amro Emad | UNIMIB 2026
</div>
