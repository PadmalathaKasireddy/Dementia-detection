# Dementia Detection

## Project Description

This project focuses on the detection of dementia using machine learning techniques. It leverages the OASIS dataset to train models that can predict the likelihood of dementia based on various features extracted from MRI scans and other demographic information.

## Dataset

The project uses two datasets from the OASIS (Open Access Series of Imaging Studies) database:
- **Cross-sectional**: A dataset that captures data at a single point in time.
- **Longitudinal**: A dataset that captures data over multiple time points for the same subjects.

## Features

The datasets include the following features:
- Subject ID
- MRI ID
- Group (Demented, Nondemented)
- Visit
- MR Delay
- Gender (M/F)
- Hand (Dominant hand)
- Age
- Education (EDUC)
- Socioeconomic Status (SES)
- Mini-Mental State Examination (MMSE)
- Clinical Dementia Rating (CDR)
- Estimated Total Intracranial Volume (eTIV)
- Normalized Whole-Brain Volume (nWBV)
- Atlas Scaling Factor (ASF)

## Getting Started

### Prerequisites

To run the project, you need the following Python libraries:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- joblib

You can install them using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn joblib
```

## Running the Project

Clone the repository:

```bash
git clone https://github.com/PadmalathaKasireddy/dementia-detection.git
cd dementia-detection
```
Load the datasets. Ensure the datasets (oasis_cross-sectional.csv and oasis_longitudinal.csv) are in the project directory.

Run the Jupyter notebook:

 ```bash
jupyter notebook Dementia_detection.ipynb
```
Follow the steps in the notebook to preprocess the data, train the models, and evaluate their performance.

## Model Training

The project includes training several machine learning models:

- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Neural Network (MLP Classifier)
- Support Vector Machine (SVM)
  
The models are evaluated based on metrics such as accuracy, ROC AUC score, and confusion matrix.

## Results

The performance of the models is summarized, and the best-performing model is identified based on the evaluation metrics.
