# Burn-Patients-Outcome-Prediction
Machine learning models and Shiraz Burn Hospital dataset for predicting burn patient outcomes (final outcome, ICU &amp; total hospitalization time, surgeries); includes evaluation code, feature-importance plots, and preprocessed Excel file.
Burn Patient Outcome Prediction Using Machine Learning

This repository contains the dataset and Python scripts used for building interpretable machine learning models to predict outcomes in burn patients. The study is based on real clinical data collected from Shiraz Burn Hospital, affiliated with Shiraz University of Medical Sciences (SUMS).

🏥 Dataset Description

The dataset includes 2725 anonymized patient records and over 133 structured variables extracted from the hospital’s burn registry. These features span:

- Demographics: Age, Gender, Nationality, Marital Status, Education
- Clinical Vitals: Blood pressure (systolic & diastolic), Pulse, Respiratory Rate, SpO₂
- Burn Characteristics: Degree, Location, and Severity
- Comorbidities: Asthma, Diabetes, Heart disease, etc.
- Treatment Info: Medication administered, ICU admission, hospital stay duration
- Outcome Labels: Final outcome, number of surgeries, ICU duration, total hospitalization

All text fields have been translated from Persian to English, and categorical variables have been numerically encoded. Missing values were imputed. All personal identifiers have been removed or anonymized.

⚖️ Legal & Ethical Compliance

The data was collected under the supervision and approval of Shiraz Burn Hospital and SUMS. Ethical clearance and consent for registry-based research were obtained. Use of this dataset in any publication or derivative work must cite the original paper and adhere to the license terms below.

🧠 Models

The following ML classifiers were applied:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Each model was evaluated for multiple target outcomes using accuracy, classification reports, confusion matrices, and ROC analysis.

📁 Contents

- `burn_data_translated.xlsx`: Cleaned and translated dataset
- `model_training.py`: Python code for training and evaluation
- `README.md`: This file
- `LICENSE`: Data and code license

📜 License

This work is distributed under the "Creative Commons Attribution 4.0 International (CC BY 4.0)" license. This means:

- You are free to "share" and "adapt" the material.
- You must give "appropriate credit" and indicate if changes were made.

📚 Citation

If you use this data or code, please cite the following:
Ali Sam. (2025). alisam1992/Burn-Patients-Outcome-Prediction: Machine learning pipeline for multiclass burn patient prognosis using a rich hospital dataset (Shiraz Burn Registry) (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.16730573

