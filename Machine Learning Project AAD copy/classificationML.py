import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

LAST_NAME = "GeminiAIAssistant"
DATA_DIR = "c:/Users/smdan/Downloads/"
RESULTS_DIR = "c:/Users/smdan/Downloads/results"

MISSING_SENTINEL = 1.0e+99
RANDOM_STATE = 42

FILES = {
    1: {"X_train": "TrainData1.txt", "y_train": "TrainLabel1.txt", "X_test": "TestData1.txt"},
    2: {"X_train": "TrainData2.txt", "y_train": "TrainLabel2.txt", "X_test": "TestData2.txt"},
    3: {"X_train": "TrainData3.txt", "y_train": "TrainLabel3.txt", "X_test": "TestData3.txt"},
    4: {"X_train": "TrainData4.txt", "y_train": "TrainLabel4.txt", "X_test": "TestData4.txt"},
}

CLASSIFIER_PIPELINE = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('scaler', StandardScaler()),
    ('model', DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10))
])

def load_data(file_map):
    def read_file(filename):
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path, header=None, sep=r'\s+', engine='python')
    Xtr = read_file(file_map['X_train'])
    ytr = read_file(file_map['y_train']).iloc[:, 0]
    Xte = read_file(file_map['X_test'])
    return Xtr.values, ytr.values, Xte.values

def preprocess(X, X_test, sentinel_value):
    X = np.where(X == sentinel_value, np.nan, X)
    X_test = np.where(X_test == sentinel_value, np.nan, X_test)
    return X, X_test

def main_classification():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for i in FILES:
        print(f"\nProcessing Dataset {i}")
        try:
            Xtr_raw, ytr, Xte_raw = load_data(FILES[i])
            Xtr, Xte = preprocess(Xtr_raw, Xte_raw, MISSING_SENTINEL)
            CLASSIFIER_PIPELINE.fit(Xtr, ytr)
            y_pred = CLASSIFIER_PIPELINE.predict(Xte).astype(int)
            out_path = os.path.join(RESULTS_DIR, f"{LAST_NAME}Classification{i}.txt")
            pd.Series(y_pred).to_csv(out_path, header=False, index=False)
            print(f"Saved predictions: {out_path}")
        except Exception as e:
            print(f"Error on Dataset {i}: {e}")

if __name__ == "__main__":
    main_classification()
