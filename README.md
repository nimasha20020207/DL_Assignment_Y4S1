# 🧠 SE4050 – Deep Learning Project

## Breast Cancer Classification Using Deep Learning

A deep learning-based classification project developed for the **SE4050 – Deep Learning** module at **SLIIT**. The project investigates different neural network approaches for classifying breast tumor cases into **malignant** and **benign** categories using structured diagnostic data.

---

## 📌 Project Overview

Breast cancer diagnosis involves analyzing various characteristics of cell nuclei to determine whether a tumor is malignant or benign. This project explores how **deep learning techniques** can be applied to structured medical data to develop an automated classification system.

The study focuses on developing, training, and evaluating multiple deep learning architectures. The models are compared using several classification metrics to identify the architecture that provides the most reliable predictive performance.

> **Important:** This project is intended as an academic decision-support study and does not replace professional medical diagnosis.

---

## 📑 Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Dataset](#-dataset)
4. [Project Objectives](#-project-objectives)
5. [Data Preparation](#-data-preparation)
6. [Deep Learning Models](#-deep-learning-models)
7. [Model Evaluation](#-model-evaluation)
8. [Results](#-results)
9. [Technology Stack](#-technology-stack)
10. [Project Workflow](#-project-workflow)
11. [Installation](#-installation)
12. [Project Structure](#-project-structure)
13. [Limitations](#-limitations)
14. [Contributors](#-contributors)

---

## 🔍 Problem Statement

Traditional diagnostic processes require medical professionals to carefully examine patient and laboratory information. Machine learning and deep learning techniques can potentially assist this process by identifying patterns within diagnostic datasets.

The main problem addressed by this project is:

> **How effectively can deep learning models classify breast tumors as malignant or benign using numerical diagnostic features?**

The project compares different neural network architectures rather than relying on a single model, allowing their strengths and weaknesses to be investigated.

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains numerical measurements describing characteristics of cell nuclei obtained from digitized breast mass images.

### Dataset Characteristics

| Property            | Description                                                                    |
| ------------------- | ------------------------------------------------------------------------------ |
| Total samples       | 569                                                                            |
| Input features      | 30 numerical features                                                          |
| Classification type | Binary classification                                                          |
| Malignant class     | M                                                                              |
| Benign class        | B                                                                              |
| Feature examples    | Radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry |

The dataset contains two target categories:

* **Malignant (M)** – cancerous tumor
* **Benign (B)** – non-cancerous tumor

Before model training, the categorical diagnosis labels are converted into numerical values suitable for binary classification.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Prepare and clean the breast cancer diagnostic dataset.
2. Perform exploratory analysis to understand the characteristics of the data.
3. Apply appropriate preprocessing and feature scaling techniques.
4. Develop multiple deep learning classification models.
5. Train and validate each architecture using a consistent experimental setup.
6. Compare the models using suitable classification metrics.
7. Analyze false-positive and false-negative predictions.
8. Identify the most suitable architecture based on experimental results.
9. Investigate whether deep learning can provide reliable classification performance on structured diagnostic data.

---

## ⚙️ Data Preparation

The dataset goes through several preprocessing stages before being provided to the neural networks.

### 1. Data Cleaning

Unnecessary attributes that do not contribute to the prediction task are removed.

Examples include:

* Patient/sample identification fields
* Empty or irrelevant columns
* Duplicate records, if identified

### 2. Target Encoding

The diagnosis variable is converted into a binary representation:

```text
Benign    → 0
Malignant → 1
```

### 3. Dataset Splitting

The dataset is divided into training, validation, and testing subsets.

A **stratified splitting strategy** is used so that both classes remain reasonably represented across the subsets.

### 4. Feature Scaling

Because the input variables have different numerical ranges, feature normalization/standardization is applied before training.

The standardization process follows:

```text
z = (x - μ) / σ
```

where:

* `x` = original feature value
* `μ` = feature mean
* `σ` = feature standard deviation

The scaler is fitted using the training data to prevent information leakage.

---

## 🧠 Deep Learning Models

To provide a meaningful comparison, several neural network architectures are investigated.

### 1️⃣ Fully Connected Neural Network

The first model acts as the baseline deep learning architecture.

A series of fully connected layers are used to learn relationships between the diagnostic features.

Example structure:

```text
Input Features
      ↓
Dense Layer
      ↓
ReLU Activation
      ↓
Dropout
      ↓
Dense Layer
      ↓
ReLU Activation
      ↓
Output Layer
      ↓
Sigmoid
```

This model provides a reference point for evaluating the more advanced architectures.

---

### 2️⃣ Residual Neural Network

A residual architecture is investigated to determine whether shortcut connections can improve learning and gradient propagation.

The general concept is:

```text
Input
  │
  ├───────────────┐
  ↓               │
Dense → BN → ReLU │
  ↓               │
Dense → BN        │
  ↓               │
  + ←─────────────┘
  ↓
Activation
```

Residual connections allow information to bypass one or more layers and can make deeper networks easier to optimize.

---

### 3️⃣ Wide & Deep Architecture

A Wide & Deep model combines two learning pathways:

* **Wide component** – captures direct relationships between features.
* **Deep component** – learns more complex nonlinear patterns.

Conceptually:

```text
                    Input Features
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
          Wide Path             Deep Path
              │              Dense → ReLU
              │              Dense → ReLU
              │              Dense → ReLU
              │                     │
              └──────────┬──────────┘
                         ↓
                     Combined
                         ↓
                    Output Layer
                         ↓
                      Sigmoid
```

This architecture is evaluated to determine whether combining shallow and deep representations improves classification performance.

---

### 4️⃣ Deep Embedded Forest

A hybrid approach combining deep learning with a traditional ensemble classifier is also investigated.

The process consists of two stages:

```text
Raw Diagnostic Features
          ↓
    Neural Network
          ↓
 Learned Feature Representation
          ↓
   Random Forest Classifier
          ↓
   Final Prediction
```

The neural network learns a compact representation of the original features, which is then provided to the Random Forest classifier.

This allows the project to compare a purely neural approach with a hybrid deep-learning and ensemble-learning approach.

---

## 📈 Model Evaluation

The models are evaluated using multiple metrics rather than accuracy alone.

### Accuracy

Measures the proportion of correctly classified samples.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

Measures how many predicted malignant cases are actually malignant.

```text
Precision = TP / (TP + FP)
```

### Recall / Sensitivity

Measures how many actual malignant cases are correctly identified.

```text
Recall = TP / (TP + FN)
```

For this application, recall is particularly important because a **false negative** represents a malignant case incorrectly classified as benign.

### F1-Score

Provides a balance between precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### ROC-AUC

The Area Under the Receiver Operating Characteristic Curve is used to assess the model's ability to distinguish between the two classes across different classification thresholds.

### Confusion Matrix

Confusion matrices are generated to examine:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## 🧪 Results

The final performance results will be added after completing model training and testing.

| Model                          | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------------------ | -------: | --------: | -----: | -------: | ------: |
| Fully Connected Neural Network |      TBD |       TBD |    TBD |      TBD |     TBD |
| Residual Neural Network        |      TBD |       TBD |    TBD |      TBD |     TBD |
| Wide & Deep Network            |      TBD |       TBD |    TBD |      TBD |     TBD |
| Deep Embedded Forest           |      TBD |       TBD |    TBD |      TBD |     TBD |

### 📌 Best Performing Model

After completing the experiments, the models will be ranked based on their test-set performance.

Special attention will be given to **malignant-class recall** and the number of **false-negative predictions**, since failing to identify a malignant case is particularly important in this application.

---

## 🔄 Project Workflow

The overall methodology can be summarized as:

```text
                Dataset
                   ↓
             Data Cleaning
                   ↓
          Exploratory Analysis
                   ↓
            Label Encoding
                   ↓
          Stratified Splitting
                   ↓
          Feature Standardization
                   ↓
       ┌───────────┼────────────┐
       ↓           ↓            ↓
      DNN       ResNet       Wide & Deep
       │           │            │
       └───────────┼────────────┘
                   ↓
          Model Evaluation
                   ↓
        Performance Comparison
                   ↓
          Best Model Selection
```

The hybrid model follows an additional feature-learning and classification stage.

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Machine Learning

* Scikit-learn

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Development Environment

* Jupyter Notebook / Google Colab
* Python virtual environment

---

## 📦 Installation

Clone the project repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis:

```bash
python cancer_analysis.py
```

Alternatively, open the provided Jupyter Notebook to execute the project step by step.

---

## 📁 Project Structure

```text
Breast-Cancer-Deep-Learning/
│
├── data/
│   └── breast_cancer_dataset.csv
│
├── notebooks/
│   └── cancer_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── training.py
│   └── evaluation.py
│
├── results/
│   ├── confusion_matrices/
│   ├── training_curves/
│   └── model_comparison.csv
│
├── requirements.txt
├── README.md
└── cancer_analysis.py
```

---

## ⚠️ Limitations

Although the models may achieve strong predictive performance, several limitations should be considered:

* The dataset contains only 569 samples.
* The data consists of structured diagnostic measurements rather than raw medical images.
* High test performance on this dataset does not guarantee performance on external clinical populations.
* The dataset may not represent all demographic and clinical variations.
* A deep learning prediction should not be interpreted as a standalone medical diagnosis.

Future work could evaluate the models on larger and more diverse datasets and investigate external validation.

---

## 🚀 Future Improvements

Potential extensions of the project include:

* Hyperparameter optimization.
* Cross-validation for more robust performance estimation.
* Class-weighting strategies.
* Explainable AI techniques such as SHAP.
* Calibration of prediction probabilities.
* External dataset validation.
* Ensemble approaches combining multiple models.
* Development of a simple web-based diagnostic-support interface.

---

## 👨‍💻 Contributors

| Student ID | Name      |
| ---------- | --------- |
| YOUR_ID    | YOUR_NAME |
| YOUR_ID    | YOUR_NAME |
| IT23153486    | NAVODYANI W M B |
| IT23322912    | WEERATHUNGA V K |

---

## 📚 Academic Context

**Module:** SE4050 – Deep Learning
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)
**Project:** Breast Cancer Classification using Deep Learning
**Academic Year:** 2026
