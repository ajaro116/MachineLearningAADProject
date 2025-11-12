import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

LAST_NAME = "GeminiAIAssistant"  
DATA_DIR = "c:/Users/smdan/Downloads/" 
RESULTS_DIR = "c:/Users/smdan/Downloads/results"

RANDOM_STATE = 42

TRAIN1 = "Copy of spam_train1.csv"
TRAIN2 = "Copy of spam_train2.csv"
TEST = "Copy of spam_test.csv"

def load_and_combine_data():
    print("Loading and combining training data...")
    
    df1 = pd.read_csv(os.path.join(DATA_DIR, TRAIN1), encoding='latin-1', header=0, 
                      usecols=['v1', 'v2'])
    df1 = df1.rename(columns={'v1': 'label', 'v2': 'text'})
    df1['label_num'] = df1['label'].map({'ham': 0, 'spam': 1})
    
    df2 = pd.read_csv(os.path.join(DATA_DIR, TRAIN2), encoding='latin-1', header=0, 
                      usecols=['text', 'label_num'])
    
    df_test = pd.read_csv(os.path.join(DATA_DIR, TEST), encoding='latin-1', header=0, 
                          usecols=['message'])
    df_test = df_test.rename(columns={'message': 'text'})
    
    df_combined = pd.concat([df1[['text', 'label_num']], df2[['text', 'label_num']]], ignore_index=True)
    
    df_combined = df_combined.dropna(subset=['text', 'label_num'])
    
    X_train = df_combined['text'].astype(str)
    y_train = df_combined['label_num'].astype(int)
    X_test = df_test['text'].astype(str)
    
    print(f"  Combined Train Samples: {len(X_train)}")
    print(f"  Test Samples: {len(X_test)}")
    print(f"  Spam/Ham distribution in combined train: {y_train.value_counts()}")
    
    return X_train, y_train, X_test

def get_candidate_pipelines():
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))
    
    return {
        'LogisticRegression': Pipeline([('tfidf', vectorizer), ('model', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))]),
        'LinearSVC': Pipeline([('tfidf', vectorizer), ('model', LinearSVC(random_state=RANDOM_STATE, dual=True))]),
        'MultinomialNB': Pipeline([('tfidf', vectorizer), ('model', MultinomialNB())]),
    }

def best_model_by_cv(X, y):
    candidates = get_candidate_pipelines()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    best_score = -1
    best_name = ''
    best_pipe = None
    
    print("\nStarting Cross-Validation (5-fold) to select the best model...")
    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_score = scores.mean()
        std_score = scores.std()
        cv_results[name] = (mean_score, std_score)
        
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_pipe = pipeline
            
    return best_name, best_pipe, cv_results

def main_spam_detection():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n{'='*60}\nProcessing Spam Email Detection\n{'='*60}")
    
    try:
        X_train, y_train, X_test = load_and_combine_data()
        
        best_name, best_pipe, cv_results = best_model_by_cv(X_train, y_train)
        
        print("\nCV Mean Accuracies:")
        for k,(m,s) in cv_results.items():
            print(f"  {k}: {m:.4f} \u00B1 {s:.4f}")
        print(f"Selected Model: {best_name}")

        print("\nFitting final model on full training data and predicting test labels...")
        best_pipe.fit(X_train, y_train)
        y_pred = best_pipe.predict(X_test)
        
        out_path = os.path.join(RESULTS_DIR, f"{LAST_NAME}SpamPrediction.txt")
        pd.Series(y_pred).to_csv(out_path, header=False, index=False)
        print(f"SUCCESS: Saved spam predictions (0=Ham, 1=Spam) to {out_path} ({len(y_pred)} samples)")

    except Exception as e:
        print(f"ERROR processing Spam Detection: {e}")
        print("Please ensure you have all data files in the DATA_DIR.")

if __name__ == '__main__':
    main_spam_detection()