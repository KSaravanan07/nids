# Network Intrusion Detection Using ML Techniques

This repository contains the implementation of network intrusion detection using machine learning techniques, focusing on dataset preprocessing, feature selection, model training, and handling class imbalance. The project is divided into two main parts: dataset exploration and anomaly detection.

## Table of Contents
- [PART-A: Datasets](#part-a-datasets)
  - [Objectives of Benchmark Datasets](#objectives-of-benchmark-datasets)
  - [Limitations of KDDCup’99 and NSL-KDD](#limitations-of-kddcup99-and-nsl-kdd)
  - [Dataset Comparison](#dataset-comparison)
- [PART-B: Anomaly Detection](#part-b-anomaly-detection)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Selection](#feature-selection)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Handling Class Imbalance](#handling-class-imbalance)
- [Deliverables](#deliverables)
- [Credits](#credits)

## PART-A: Datasets

### Objectives of Benchmark Datasets
- We explore the objectives behind creating benchmark datasets for intrusion detection, including providing diverse attack types, realistic traffic patterns, and balanced data for training and testing algorithms.

### Limitations of KDDCup’99 and NSL-KDD
- We analyze the drawbacks of the KDDCup’99 dataset, such as redundant records and class imbalance, which led to the development of NSL-KDD. Both datasets are considered unreliable for validating novel intrusion detection algorithms due to their outdated attack types and lack of representation of modern threats. [Read more](https://ieeexplore.ieee.org/document/8586840).

### Dataset Comparison
- A comparative table is provided for the datasets: KDD CUP’99, NSL-KDD, CICIDS 2017, CICIDS 2018, and UNSW-NB15. The comparison includes properties such as year of public availability, number of features, class labels, and types of attacks.

## PART-B: Anomaly Detection

### Data Preprocessing
- The CICIDS2017 and UNSW-NB15 partial datasets are used for anomaly detection. We cleaned the data by removing source IP, destination IP, source port, destination port, and timestamp fields to prevent bias in detection algorithms. 
- We unified each dataset by concatenating the corresponding CSV files, re-labeled attack labels as '1', and benign samples as '0'. 
- Categorical features were encoded using label encoding, and columns with over 25% missing values were dropped. Remaining missing values were filled with the column mean or the most frequent value.
- Duplicate and multi-label rows were removed, and the data was normalized using a min-max scaler.

### Feature Selection
- We used the SelectKBest method with chi-square scoring for univariate feature selection, testing with three different values of 'k'. This resulted in three variants of refined datasets for both CICIDS2017 and UNSW-NB15.
- As an optional extension, Principal Component Analysis (PCA) was used to reduce dimensionality, capturing at least 90% of the variance.

### Model Training and Evaluation
- The refined datasets were split into training, validation, and test sets (80:20 split). Various models were trained, including:
  - Gaussian Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM) with a linear kernel
  - Decision Trees (ID3, C4.5)
  - Random Forest
  - XGBoost
  - Voting Classifier (combining Gaussian Naive Bayes, Logistic Regression, SVM with a linear kernel)
  - AdaBoost Classifier
  - MLP Classifier with one hidden layer of size 100.
- Confusion matrices were created for each model, and a comparison table was compiled showing accuracy, precision, recall, F1-score, and runtime for each model. AUC-ROC curves were plotted for all models.

### Handling Class Imbalance
- We tackled class imbalance using data-driven resampling techniques:
  - Random OverSampling (ROS)
  - Random UnderSampling (RUS)
  - Synthetic Minority Oversampling Technique (SMOTE)
- The models were re-evaluated using the balanced datasets from the above resampling techniques.

## Deliverables
- The following files are included in the repository:
  - A report for PART-A detailing the objectives, limitations, and comparisons of the datasets.
  - A Python notebook for PART-B containing all plots, confusion matrices, and comparison tables. The notebook is executable on Google Colab.
  - A credit statement document providing a detailed description of each group member’s contributions to the project, including coding, plots, report writing, bug fixes, and analysis.

## Credits
- This project was completed as a collaborative effort. The credit statement outlines each group member's contributions to ensure transparency and acknowledgment of individual efforts.

