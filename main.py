import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)

# === Load Excel Data ===
df = pd.read_excel("burn_data_translated.xlsx")

# === Clean Unknown Values ===
unknown_vals = ["Unknown", "Not specified", "--", "-", "نامشخص", "N/A", "None", "No data", "NaN", "missing", "null", "NULL"]
df.replace(unknown_vals, 0, inplace=True)

# === Map String Columns to Integers ===
value_maps = {}
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        unique_vals = df[col].dropna().unique()
        str_vals = [val for val in unique_vals if isinstance(val, str)]
        val_map = {val: i + 1 for i, val in enumerate(sorted(str_vals))}
        value_maps[col] = val_map
        df[col] = df[col].map(val_map).where(df[col].notna(), df[col])
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

# === Final NaN fill ===
df.fillna(0, inplace=True)
df = df.apply(pd.to_numeric, errors='ignore')

# === Configuration ===
target_column = 'Final Outcome'  # change here as needed
id_column = 'ID'
q_bins = 4

# === Check Required Columns ===
if id_column not in df.columns or target_column not in df.columns:
    raise ValueError("❌ Required columns missing in data.")

# === Conditional Target Preparation ===
bin_target = target_column in ['Total Hospitalization Time', 'ICU Hospitalization Time']
if bin_target:
    df_zero = df[df[target_column] == 0].copy()
    df_nonzero = df[df[target_column] > 0].copy()

    bin_labels = [2, 3, 4, 5]
    df_nonzero['Hosp_Bin'] = pd.qcut(df_nonzero[target_column], q=q_bins, labels=bin_labels, duplicates='drop')
    df_zero['Hosp_Bin'] = 1
    df_binned = pd.concat([df_zero, df_nonzero], ignore_index=True)
    df_binned.dropna(subset=['Hosp_Bin'], inplace=True)
    y = df_binned['Hosp_Bin'].astype(int)
    X = df_binned.drop(columns=[id_column, target_column, 'Hosp_Bin'])
else:
    df_cleaned = df.dropna(subset=[target_column])
    y = df_cleaned[target_column].astype(int)
    X = df_cleaned.drop(columns=[id_column, target_column])

# === Imputation & Train/Test Split ===
X = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Define Classifiers ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

# === Train and Evaluate Models ===
best_rf_model = None
for name, model in models.items():
    print(f"\n🔹 {name}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.grid(False)
    plt.show()

    if hasattr(model, "predict_proba"):
        y_bin = label_binarize(y_test, classes=np.unique(y))
        y_score = pipeline.predict_proba(X_test)

        for i in range(y_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            plt.plot(fpr, tpr, label=f'Class {i+1} (AUC={auc(fpr, tpr):.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.grid(True)
        plt.show()

    if name == "Random Forest":
        best_rf_model = model

# === Feature Importance (Top 10) from Random Forest ===
if best_rf_model:
    best_rf_model.fit(X_train, y_train)
    importances = pd.Series(best_rf_model.feature_importances_, index=X.columns)
    top_10 = importances.sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top_10.index[::-1], top_10.values[::-1], color='darkcyan')
    plt.xlabel("Importance Score")
    plt.title("Top 10 Important Features (Random Forest)")
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.show()
