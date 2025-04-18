# AIDI1009-24F-10827: Assignment #3 - AI-Driven Brain Tumor Classification Using ResNet50 + MLJAR AutoML

## Group Members:
- Ranveer Singh Saini - 200569800
- Girik Nohani - 200565756
- Pooja Indraj Yadav - 200568689
- Ronald Kalani - 200619730

**Instructor:** Jahanzeb Abbas  

**Course:** Technology & Visual Arts - AIDI 1010 - Emerging Technologies  

**Submission Date:** April 19, 2025

---

## Project Overview
This project focuses on building an AI-powered diagnostic system to classify brain tumor types from MRI images. It leverages a hybrid modeling approach: ResNet50 for deep image-based feature extraction and MLJAR AutoML for structured data classification. This approach aims to improve diagnostic accuracy, minimize human error, and reduce the time required for manual diagnosis, offering a scalable solution for healthcare applications.


## Goal
The primary goal is to create an automated classification system for brain tumors visible in MRI scans. The hybrid solution involves using ResNet50, a CNN pre-trained on ImageNet, for image feature extraction, followed by MLJAR AutoML to handle model training, optimization, and evaluation. This architecture improves model accuracy, generalization, and robustness.


## Intended Audience
This project is aimed at:
- Clinical researchers and radiologists
- AI/ML engineers focused on healthcare
- Medical educators and students
- Hospital administrators and MedTech decision-makers
- 

## Strategy & Pipeline Steps
1. **Preprocessing:** MRI images were resized to 224x224 pixels and normalized.
2. **Transfer Learning:** Features extracted using ResNet50 with frozen early layers.
3. **Dataset Splitting:** Stratified sampling into training (80%) and validation (20%) sets.
4. **AutoML Modeling:** Features reshaped and fed into MLJAR AutoML for classification.
5. **Evaluation:** Results measured using accuracy, precision, recall, F1-score, and ROC-AUC.
6. **Comparison:** Benchmarked against Assignment 2 prototype and published research.
7. 

## Challenges
- **Class imbalance** resolved via augmentation (rotation, flipping).
- **Limited dataset size** addressed through transfer learning.
- **Lack of GPU** mitigated by optimizing batch sizes and freezing layers.

## Problem Statement
Can an AI system accurately classify MRI-based brain tumor images into three categories (Glioma, Meningioma, and Pituitary tumor), and support radiologists by improving consistency and diagnostic speed?


## Dataset
The dataset used was sourced from Kaggle and contains T1-weighted contrast-enhanced MRI images divided into four classes (Glioma, Meningioma, Pituitary Tumor, Normal). All images were organized into structured subfolders for training and validation via Google Drive.


## Implementation Overview
- Mounted Google Drive to load images.
- Unified datasets into one folder with four subcategories.
- Visualized samples for each tumor type.
- Used `ImageDataGenerator` for 80/20 train-validation split.
- Constructed ResNet50-based CNN model.
- Trained for 5 epochs (accuracy: ~63%, val_acc: ~64%).
- Evaluated with confusion matrix and classification report (F1-score: 0.32 avg).


## MLJAR AutoML Pipeline
- ResNet50 features flattened and reshaped.
- MLJAR AutoML initiated in 'Compete' mode.
- Best model: Random Forest (logloss: 0.63, accuracy: ~88%).
- AutoML steps included ensembling and stacking.
- 

## Visualizations & Results
- Confusion matrix and heatmaps used.
- Accuracy/loss plotted across epochs.
- Visual tools presented for executive review.
- Comparison clearly highlighted improvements from 26% (Assignment 2) to ~88% (AutoML pipeline).
- 

## Conceptual Enhancement
**Artificial General Intelligence (AGI)** is proposed as an enhancement. Future diagnostic systems should adapt autonomously, interpret unseen MRI patterns, and reason across institutions. AGI aligns with this vision, surpassing the capabilities of narrow AI systems.


## Lessons Learned
- AutoML saves time and improves performance.
- Transfer learning with ResNet50 is efficient for small medical datasets.
- More epochs and feature selection will further improve accuracy.
- 

## Group Member Contributions
- **Ranveer Singh Saini:** Led MLJAR integration and GitHub documentation.
- **Girik Nohani:** Focused on model evaluation and research alignment.
- **Pooja Indraj Yadav:** Pioneered Grad-CAM visualizations and structure.
- **Ronald Kalani:** Led visual design, executive output formatting, and classification reporting.
- 

## References
1. Abdusalomov, A. B. et al. (2023). Brain tumor detection using DL. *Cancers*.
2. Cheng, J. et al. (2015). Tumor region augmentation in CNNs. *PLOS ONE*.
3. Menze, B. H. et al. (2015). BRATS benchmark. *IEEE Trans. Med Imaging*.
4. MLJAR AutoML Documentation - https://github.com/mljar/mljar-supervised
5. Kaggle Brain MRI Dataset - https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

---

## Repository Structure
- `/reports` – Executive summaries & presentation PDFs
- `/visualizations` – Confusion matrices, training curves, heatmaps
- `README.md` – This file
- `Emerging_Technology_Group_Assignment_3.ipynb` – Main implementation notebook

- 

### 📎 GitHub Repository:
[https://github.com/ronaldkalani/Emerging-Technology-Group-Assignment-3](https://github.com/ronaldkalani/Emerging-Technology-Group-Assignment-3)


