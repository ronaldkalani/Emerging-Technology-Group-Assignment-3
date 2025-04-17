# AI-Driven Brain Tumor Classification Using ResNet50 + MLJAR AutoML

**Course:** AIDI 1010 - Emerging Technologies  
**Semester:** Winter 2025  
**Assignment #3 – Group Project**  
**Submission Date:** April 19, 2025

## Group Members & Instructor

- **Ranveer Singh Saini** – 200569800  
- **Girik Nohani** – 200565756  
- **Pooja Indraj Yadav** – 200568689  
- **Ronald Kalani** – 200619730  
- **Instructor:** Jahanzeb Abbas

---

## Problem Statement

Brain tumor detection using MRI images is essential for early diagnosis and treatment planning. Manual analysis is time-consuming and subject to interpretation variability among radiologists. This project aims to develop an AI-powered solution that automates tumor classification using deep learning and AutoML.

---

## Goal

To build a hybrid AI model combining **ResNet50 (for image-based feature extraction)** and **MLJAR AutoML (for structured learning and automated model tuning)** to classify MRI brain scans into four categories:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

---

## Intended Audience

This project is designed for:
- **Radiologists** and **medical professionals** seeking AI-assisted diagnostics
- **AI researchers** and **students** exploring hybrid modeling strategies
- **Developers** building intelligent healthcare platforms

---

## Strategy & Pipeline

1. **Preprocessing:** Resize MRI images to 224x224, normalize pixel values.
2. **Feature Extraction:** Use pre-trained **ResNet50 (with frozen weights)** to extract image embeddings.
3. **Classification:** Train Random Forest and other models using **MLJAR AutoML** on extracted features.
4. **Explainability:** Apply Grad-CAM to interpret model predictions visually.
5. **Evaluation:** Accuracy, precision, confusion matrix, and Grad-CAM overlays.

---

##  Challenges Faced

- **Dataset imbalance** across tumor types.
- **Dependency conflicts** between MLJAR and TensorFlow libraries.
- **GPU limitations** in local environment—resolved using **Google Colab**.
- Adjusting **label encoding** for compatibility with AutoML models.

---

##  Dataset

The MRI dataset includes four well-structured folders:

- `/Normal`
- `/glioma_tumor`
- `/meningioma_tumor`
- `/pituitary_tumor`

These were merged into a common dataset with proper subdirectory labels and loaded using `ImageDataGenerator` with an 80-20 split.

---

## Steps (Code Snippets)

```python
# Load and preprocess images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory(data_path, target_size=(224,224), class_mode='categorical', subset='training')

# Load ResNet50 with frozen base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

