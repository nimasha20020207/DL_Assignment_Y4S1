<div align="center">

# 🧠 SE4050 – Deep Learning Group Project

## 🎗️ Breast Cancer Classification Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

**Group ID:** `SE4050_G10` · **Academic Year:** `Y4S1 – 2026`

</div>

This project was completed for the **SE4050 – Deep Learning** module at the **Sri Lanka Institute of Information Technology (SLIIT)**. It compares four supervised deep-learning architectures for classifying breast tumour records as either **benign** or **malignant** using the Breast Cancer Wisconsin Diagnostic dataset.

> **Important:** This is an academic experiment and is not intended to replace diagnosis or decisions made by qualified medical professionals.

---

## 📑 Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Objectives](#project-objectives)
- [Data Preparation](#data-preparation)
- [Model Architectures](#model-architectures)
- [Model Evaluation](#model-evaluation)
- [Final Results](#final-results)
- [Project Workflow](#project-workflow)
- [Technology Stack](#technology-stack)
- [Running the Project](#running-the-project)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)

---

## 📌 Project Overview

Breast-cancer classification requires several measurements describing the size, shape and texture of cell nuclei to be considered together. This project investigates how different neural-network architectures learn from those numerical measurements and whether more advanced architectures provide a meaningful improvement over simpler models.

The four models were developed and compared using a consistent workflow that included dataset inspection, exploratory data analysis, feature standardisation, stratified data splitting, model training and test-set evaluation. Performance was assessed using accuracy, precision, recall, F1-score, ROC-AUC, learning curves, classification reports and confusion matrices.

## 🔍 Problem Statement

The central problem addressed by the project is:

> How can four distinct supervised deep-learning architectures be developed and fairly compared for benign and malignant breast-tumour classification using consistent preprocessing and comprehensive evaluation metrics?

False-negative predictions are especially important because they represent malignant cases incorrectly classified as benign. Therefore, the comparison considers recall and confusion-matrix results in addition to overall accuracy.

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic dataset**, which contains measurements obtained from digitised images of fine-needle aspirate samples of breast masses.

| Property | Description |
| --- | --- |
| Total records | 569 |
| Original columns | 33 |
| Model input features | 30 numerical features |
| Benign records | 357 |
| Malignant records | 212 |
| Classification task | Binary classification |
| Target encoding | Benign = 0, Malignant = 1 |

The features describe ten main cell-nucleus characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each characteristic is represented using mean, standard-error and worst-case measurements.

### Class Distribution

```text
Benign (B)    ███████████████████████████████  357 (62.74%)
Malignant (M) ██████████████████               212 (37.26%)
```

## 🎯 Project Objectives

1. Inspect and clean the selected breast-cancer dataset.
2. Explore the class distribution and relationships among the diagnostic features.
3. Encode the diagnosis labels and standardise the numerical input features.
4. Divide the data into stratified training, validation and test sets.
5. Prevent data leakage by fitting preprocessing operations only on training data.
6. Develop and train four different supervised learning architectures.
7. Evaluate the models using multiple classification metrics and visualisations.
8. Compare predictive performance, learning behaviour and model complexity.
9. Identify the best-performing model and discuss the limitations of the experiment.

## ⚙️ Data Preparation

The following preprocessing steps were applied before model development:

1. The completely empty `Unnamed: 32` column was removed.
2. The `id` column was excluded because it does not describe a medical characteristic of the tumour.
3. Diagnosis labels were encoded as `B = 0` and `M = 1`.
4. Stratified splitting was used to preserve similar class proportions across the data subsets.
5. The 30 numerical features were standardised using `StandardScaler`.
6. The scaler was fitted only on the training set and then applied to the validation and test sets to reduce data leakage.

The common experimental split described in the report contains 397 training records, 86 validation records and 86 test records.

## 🧠 Model Architectures

### 1️⃣ Model 1 – Standard Multi-Layer Perceptron (MLP)

The baseline model uses 30 standardised inputs, two fully connected hidden layers with 64 and 32 neurons, ReLU activation and a sigmoid output layer. It provides a relatively simple and computationally efficient reference model.

### 2️⃣ Model 2 – Deep Embedded Forest (DEF)

This hybrid architecture uses a neural network to transform the original features into a compact 16-dimensional embedding. The learned representation is then supplied to a Random Forest classifier for the final prediction.

### 3️⃣ Model 3 – Pre-Activation Residual Neural Network

The Pre-Activation ResNet uses batch normalisation and ReLU activation before dense transformations. Shortcut connections allow information to pass directly through residual blocks, while dropout, early stopping and learning-rate reduction support stable training. The model contains 25,377 parameters.

### 4️⃣ Model 4 – Wide & Deep Neural Network

This architecture contains two parallel paths. The wide branch learns direct relationships between the input features and the output, while the deep branch uses dense layers of 128, 64 and 32 units to learn nonlinear relationships. The two outputs are combined before the final sigmoid activation.

## 📈 Model Evaluation

The following measures were used to evaluate and compare the models:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification report
- Confusion matrix
- ROC curve
- Training and validation learning curves
- Training time and model complexity

## 🧪 Final Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard MLP | 98.84% | 98.86% | 98.84% | 98.83% | 98.29% |
| Deep Embedded Forest | 98.25% | 98.00% | 98.00% | 98.00% | Not reported |
| Wide & Deep Network | 97.67% | 100.00% | 93.75% | 96.77% | 100.00% |
| Pre-Activation ResNet | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

> **Result summary:** All four models achieved more than 97% test accuracy. The Pre-Activation ResNet produced the highest result on its selected test split, while the Standard MLP provided an excellent balance between performance and simplicity.

### 🏆 Best-Performing Model

The **Pre-Activation ResNet** achieved the highest results across all reported evaluation metrics. It correctly classified all 86 records in its selected test split, including 54 benign and 32 malignant records.

These perfect scores must still be interpreted carefully. The dataset contains only 569 records, and the result was obtained from a single test split. It does not guarantee equivalent performance on independent real-world clinical data.

The **Standard MLP** also produced strong results and provides an attractive balance between predictive performance, computational efficiency and ease of implementation.

## 🔄 Project Workflow

```text
Dataset Collection and Inspection
              ↓
Exploratory Data Analysis
              ↓
Data Cleaning and Target Encoding
              ↓
Stratified Train/Validation/Test Split
              ↓
Training-Only Feature Standardisation
              ↓
Development and Training of Four Models
              ↓
Test-Set Evaluation
              ↓
Performance Comparison and Best-Model Selection
```

## 🛠️ Technology Stack

- **Programming language:** Python
- **Deep learning:** TensorFlow, Keras
- **Machine learning and evaluation:** Scikit-learn
- **Data processing:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Development environment:** Google Colab / Jupyter Notebook
- **Version control:** Git and GitHub

## 🚀 Running the Project

Clone the repository:

```bash
git clone https://github.com/nimasha20020207/DL_Assignment_Y4S1.git
cd DL_Assignment_Y4S1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook for the required model in Google Colab or Jupyter Notebook and run its cells from top to bottom. Ensure that `Cancer_Data.csv` is available in the notebook's working directory.

## 📁 Repository Structure

```text
DL_Assignment_Y4S1/
├── Cancer_Data.csv
├── IT23259584.ipynb    # Model 1 – Standard MLP
├── IT23309142.ipynb    # Model 2 – Deep Embedded Forest
├── IT23153486.ipynb    # Model 3 – Pre-Activation ResNet
├── IT23322912.ipynb    # Model 4 – Wide & Deep Network
├── requirements.txt
└── README.md
```

## ⚠️ Limitations

- The dataset contains only 569 records.
- The models were evaluated using a single data split.
- The data consists of structured measurements rather than raw medical images.
- Results may vary with a different random split or execution environment.
- The perfect ResNet result may be optimistic because of the small test set.
- Performance on independent clinical datasets has not been evaluated.
- The predictions must not be treated as standalone medical diagnoses.

## 💡 Future Improvements

- Repeated stratified evaluation or k-fold cross-validation
- Hyperparameter optimisation
- Evaluation using larger independent datasets
- Classification-threshold optimisation to reduce false negatives
- Explainable AI methods such as SHAP
- Probability calibration and model uncertainty analysis

## 👥 Contributors

| Student ID | Student Name | Contribution |
| --- | --- | --- |
| IT23259584 | Karunarathne K.D.N.S. | Model 1 – Standard Multi-Layer Perceptron |
| IT23309142 | Mihiranga U.G.P.G. | Model 2 – Deep Embedded Forest |
| IT23153486 | Navodyani W.M.B. | Model 3 – Pre-Activation Residual Neural Network |
| IT23322912 | Weerathunga V.K. | Model 4 – Wide & Deep Neural Network |

## 📤 Submission Notes

Before submitting the project to Gradescope:

- Run every notebook from top to bottom without errors.
- Keep the final cell outputs, metrics and graphs saved.
- Include all four member notebooks, `Cancer_Data.csv`, `README.md` and `requirements.txt`.
- Confirm that the filenames match the student IDs.
- Follow the lecturer's instructions on whether to upload individual files or a single ZIP archive.

## 🎓 Academic Context

- **Module:** SE4050 – Deep Learning
- **Institution:** Sri Lanka Institute of Information Technology (SLIIT)
- **Group ID:** SE4050_G10
- **Academic Year:** Y4S1 – 2026
- **Repository:** <https://github.com/nimasha20020207/DL_Assignment_Y4S1.git>

## 📚 References

1. [UCI Machine Learning Repository – Breast Cancer Wisconsin Diagnostic Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
2. [TensorFlow Documentation](https://www.tensorflow.org/)
3. [Keras Documentation](https://keras.io/)
4. [Scikit-learn Documentation](https://scikit-learn.org/)
