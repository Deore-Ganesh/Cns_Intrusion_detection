"""
CLI Pipeline Runner for:
Evaluation of Machine Learning Algorithms in Network-Based Intrusion Detection System
"""

import os
import sys
import argparse
from time import process_time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

def get_models():
    return {
        'Decision Tree': tree.DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.000138),
        'Random Forest': RandomForestClassifier(n_jobs=-1, criterion='gini', max_depth=10, min_samples_leaf=0.00005, min_samples_split=12, n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(C=10, gamma='scale', kernel='rbf', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(var_smoothing=8.11e-06),
        'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(40,), activation='relu', solver='adam', alpha=0.001, max_iter=250, random_state=42),
        'Deep Neural Network': MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation='tanh', solver='adam', alpha=1e-05, max_iter=250, random_state=42)
    }

def run_dataset_evaluation(dataset_type='luflow'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n" + "="*80)
    print(f" RUNNING EVALUATION PIPELINE: {dataset_type.upper()} DATASET ")
    print("="*80)

    if dataset_type == 'luflow':
        fs_file = os.path.join(base_dir, 'Dataset', 'features_selected', 'LUFlow_RandomForestClassifier_11.csv')
        train_file = os.path.join(base_dir, 'Dataset', 'dataset_cleaned', 'LUFlow.csv')
        test_file = os.path.join(base_dir, 'Dataset', 'dataset_cleaned', 'LUFlow2021.csv')
        n_features = 6
        train_label_name = 'LUFlow 2020 (In-Dist)'
        test_label_name = 'LUFlow 2021 (Progressive/Out-of-Dist)'
    else:
        fs_file = os.path.join(base_dir, 'Dataset', 'features_selected', 'CIC-IDS2017_RandomForestClassifier_20.csv')
        train_file = os.path.join(base_dir, 'Dataset', 'dataset_cleaned', 'CIC-IDS2017.csv')
        test_file = os.path.join(base_dir, 'Dataset', 'dataset_cleaned', 'CSE-CIC-IDS2018.csv')
        n_features = 11
        train_label_name = 'CIC-IDS2017 (In-Dist)'
        test_label_name = 'CSE-CIC-IDS2018 (Progressive/Out-of-Dist)'

    if not os.path.exists(fs_file) or not os.path.exists(train_file):
        print(f"[!] Required files missing. Generating benchmark data first...")
        from generate_benchmark_data import generate_benchmark_data
        generate_benchmark_data(base_dir)

    features = pd.read_csv(fs_file, header=None).squeeze()
    if isinstance(features, pd.DataFrame):
        features = features.iloc[:, 0]
    selected_features = list(features[:n_features])
    print(f"[*] Selected Top {n_features} Features:")
    for idx, f in enumerate(selected_features, 1):
        print(f"    {idx}. {f}")

    cols = selected_features + ['label']
    df_train = pd.read_csv(train_file, usecols=cols)
    df_test = pd.read_csv(test_file, usecols=cols)
    print(f"[*] Training samples: {len(df_train)}, Progressive testing samples: {len(df_test)}")

    # Split train into train & in-distribution validation
    X = df_train.drop('label', axis=1)
    y = df_train['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    results = []

    print("\n" + "-"*80)
    print(f"{'Model':<26} | {'Train Sec':<9} | {'In-Dist Acc':<11} | {'In-Dist F1':<10} | {'Prog. Acc':<10} | {'Prog. F1':<9}")
    print("-"*80)

    for name, clf in models.items():
        t0 = process_time()
        clf.fit(X_train_scaled, y_train)
        train_time = process_time() - t0

        # Evaluate In-Distribution
        preds_val = clf.predict(X_val_scaled)
        acc_val = metrics.accuracy_score(y_val, preds_val)
        f1_val = metrics.f1_score(y_val, preds_val, pos_label='malicious', zero_division=0)

        # Evaluate Progressive Dataset
        preds_test = clf.predict(X_test_scaled)
        acc_test = metrics.accuracy_score(y_test, preds_test)
        f1_test = metrics.f1_score(y_test, preds_test, pos_label='malicious', zero_division=0)

        print(f"{name:<26} | {train_time:>8.3f}s | {acc_val*100:>10.2f}% | {f1_val:>10.4f} | {acc_test*100:>9.2f}% | {f1_test:>9.4f}")
        results.append({
            'Model': name,
            'Train_Time_s': train_time,
            'In_Dist_Acc': acc_val,
            'In_Dist_F1': f1_val,
            'Prog_Acc': acc_test,
            'Prog_F1': f1_test
        })

    print("-"*80 + "\n")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run Intrusion Detection ML Pipeline")
    parser.add_argument('--dataset', choices=['luflow', 'cic', 'all'], default='all', help="Which dataset to evaluate (default: all)")
    args = parser.parse_args()

    if args.dataset in ['luflow', 'all']:
        run_dataset_evaluation('luflow')
    if args.dataset in ['cic', 'all']:
        run_dataset_evaluation('cic')

if __name__ == '__main__':
    main()
