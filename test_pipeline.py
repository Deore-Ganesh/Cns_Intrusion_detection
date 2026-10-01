import os
import pandas as pd
import numpy as np
from time import process_time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

def test_pipeline():
    print("Testing LUFlow evaluation pipeline...")
    features = pd.read_csv('Dataset/features_selected/LUFlow_RandomForestClassifier_11.csv', header=None).squeeze()
    features = features[:6]
    print(f"Top 6 features: {list(features)}")
    
    cols = features.tolist() + ['label']
    luflow_2020 = pd.read_csv('Dataset/dataset_cleaned/LUFlow.csv', usecols=cols)
    luflow_2021 = pd.read_csv('Dataset/dataset_cleaned/LUFlow2021.csv', usecols=cols)
    print(f"Loaded train 2020: {luflow_2020.shape}, test 2021: {luflow_2021.shape}")
    
    # Train / test split
    train_X, test_X, train_y, test_y = train_test_split(
        luflow_2020.drop('label', axis=1), 
        luflow_2020['label'], 
        test_size=0.2, 
        random_state=10
    )
    
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)
    test2021_X_scaled = scaler.transform(luflow_2021.drop('label', axis=1))
    test2021_y = luflow_2021['label']
    
    models = {
        'Decision Tree': tree.DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.000138),
        'Random Forest': RandomForestClassifier(n_jobs=-1, criterion='gini', max_depth=10, min_samples_leaf=0.00005, min_samples_split=12, n_estimators=50, random_state=10),
        'Support Vector Machine': SVC(C=10, gamma='scale', kernel='rbf', random_state=10),
        'Naive Bayes': GaussianNB(var_smoothing=8.11e-06),
        'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(40,), activation='relu', solver='adam', alpha=0.001, max_iter=200, random_state=10)
    }
    
    for name, clf in models.items():
        t0 = process_time()
        clf.fit(train_X_scaled, train_y)
        train_time = process_time() - t0
        
        preds_in = clf.predict(test_X_scaled)
        acc_in = metrics.accuracy_score(test_y, preds_in)
        f1_in = metrics.f1_score(test_y, preds_in, pos_label='malicious')
        
        preds_out = clf.predict(test2021_X_scaled)
        acc_out = metrics.accuracy_score(test2021_y, preds_out)
        f1_out = metrics.f1_score(test2021_y, preds_out, pos_label='malicious')
        
        print(f"[{name}] Train Time: {train_time:.2f}s | In-Dist (2020) Acc: {acc_in:.4f} | Out-of-Dist (2021) Acc: {acc_out:.4f}")

if __name__ == '__main__':
    test_pipeline()
