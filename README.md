# 📦 GhostNet-Small: Efficient CNNs & Distillation Strategies for Tiny Images

This repository contains the code and experiments from the paper:
**“GhostNet-Small: A Tailored Architecture and Comparative Study of Distillation Strategies for Tiny Images”**
by *Florian Zager (Karlsruhe Institute of Technology)*

## 🔍 Overview

Modern neural networks often achieve state-of-the-art accuracy by increasing model size—but this comes at a cost: they are often too large for real-time inference on edge devices. This project investigates both architectural compression and knowledge distillation (KD) strategies to enable efficient CNNs for **low-resolution datasets** such as **CIFAR-10**.

### Contributions

* 🔧 **GhostNet-Small**: A modified version of GhostNetV3 tailored for low-resolution (32×32) images.
* 🧪 **Comparative evaluation** of knowledge distillation strategies:

  * Traditional Distillation
  * Teacher Assistant Distillation
  * Teacher Ensemble Distillation
* 📊 Analysis of performance trade-offs between architecture compression and various KD techniques.

## 🏗️ Architecture

GhostNet-Small is a parameterized lightweight CNN based on GhostNetV3. It reduces model complexity while improving accuracy on small inputs. Width multipliers allow scaling the model from ultra-compact to near teacher-level performance.

## 🧪 Experiments

### 🔬 Evaluated Techniques

| Technique            | Result (∆ Accuracy)        |
| -------------------- | -------------------------- |
| **Baseline (no KD)** | ✅ Highest accuracy overall |
| KD                   | 🔻 Decrease                |
| Teacher Assistant    | 🔻 Decrease                |
| Teacher Ensemble     | 🔻 Decrease.               |

### ⚠️ Key Finding

> All evaluated distillation techniques **underperformed** compared to baseline training, suggesting architectural adaptation (GhostNet-Small) may be **more impactful** than KD for low-resolution data.

## 📊 Results Summary

| Model           | Params (M) | Top-1 Acc (%) |
| --------------- | ---------- | ------------- |
| GN-S 1.0x       | 0.48       | 91.37         |
| GN-S 2.8x       | 2.42       | **93.94**     |
| GN-D (Original) | 6.86       | 91.23         |
| ResNet-50       | 23.5       | 93.65         |
| EfficientNetV2  | 20.1       | **97.90**     |
