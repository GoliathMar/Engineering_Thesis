# Comparative Analysis of Image Interpolation Methods

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Thesis](https://img.shields.io/badge/Type-Engineering%20Thesis-green)
![Status](https://img.shields.io/badge/Status-Active-orange)

## 📄 About The Project

This repository contains the source code for my **Engineering Thesis**. The main goal of this project is to perform a comparative analysis of various image interpolation and super-resolution techniques.

The project focuses on upscaling images (specifically optimized for **2x upscaling**) and comparing the quality of reconstruction between traditional algorithmic methods and modern Deep Learning approaches.

## 🧠 Implemented Methods

The project implements and compares the following 5 methods:

### Traditional Methods
* **Nearest Neighbour** - The simplest interpolation method.
* **Bilinear** - Weighted average of the 4 nearest pixels.
* **Bicubic** - More complex, using 16 nearest pixels for smoother results.

### Deep Learning Methods (Super-Resolution)
* **SRCNN** (Super-Resolution Convolutional Neural Network) - An end-to-end deep learning network for image super-resolution.
* **SRGAN** (Super-Resolution Generative Adversarial Network) - Uses a GAN framework to infer photo-realistic natural images for 2x upscaling factors

## 💾 Dataset

For training and inference purposes, the **DIV2K** dataset was utilized. This dataset is a standard benchmark for image super-resolution tasks, containing high-quality images of diverse resolutions.

## 🚀 Getting Started

### Prerequisites
* Python 3.12
* PyCharm (Recommended)
* Libraries found in requirements.txt

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/GoliathMar/Engineering_Thesis.git](https://github.com/GoliathMar/Engineering_Thesis.git)
   pip install -r requirements.txt
