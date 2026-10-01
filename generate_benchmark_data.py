import os
import pandas as pd
import numpy as np

def generate_benchmark_data(base_path):
    # 1. Feature selection files
    fs_dir = os.path.join(base_path, 'Dataset', 'features_selected')
    os.makedirs(fs_dir, exist_ok=True)
    
    cic_features = [
        "Bwd Packet Length Std",
        "Average Packet Size",
        "Max Packet Length",
        "Packet Length Variance",
        "Packet Length Std",
        "Avg Bwd Segment Size",
        "Packet Length Mean",
        "Destination Port",
        "Init_Win_bytes_forward",
        "Fwd Packet Length Mean",
        "Init_Win_bytes_backward",
        "Subflow Bwd Bytes",
        "Bwd Packet Length Max",
        "Total Length of Bwd Packets",
        "Bwd Packet Length Mean",
        "Fwd Packet Length Max",
        "Bwd Packet Length Min",
        "Subflow Fwd Bytes",
        "Avg Fwd Segment Size",
        "Fwd Header Length"
    ]
    pd.Series(cic_features).to_csv(os.path.join(fs_dir, 'CIC-IDS2017_RandomForestClassifier_20.csv'), index=False, header=False)
    
    luflow_features = [
        "dest_port",
        "bytes_out",
        "total_entropy",
        "src_port",
        "num_pkts_in",
        "duration",
        "avg_ipt",
        "entropy",
        "bytes_in",
        "num_pkts_out",
        "proto"
    ]
    pd.Series(luflow_features).to_csv(os.path.join(fs_dir, 'LUFlow_RandomForestClassifier_11.csv'), index=False, header=False)
    
    # 2. Cleaned datasets directory
    clean_dir = os.path.join(base_path, 'Dataset', 'dataset_cleaned')
    os.makedirs(clean_dir, exist_ok=True)
    
    np.random.seed(42)
    n_samples = 3000
    
    # Generate LUFlow 2020 (Training) and LUFlow 2021 (Testing)
    for year, name in [('2020', 'LUFlow.csv'), ('2021', 'LUFlow2021.csv')]:
        labels = np.random.choice(['benign', 'malicious'], size=n_samples, p=[0.5, 0.5])
        data = {
            'dest_port': np.where(labels == 'benign', np.random.choice([80, 443, 22, 53, 8080], size=n_samples), np.random.choice([4444, 1337, 21, 23, 3389, 80], size=n_samples)),
            'bytes_out': np.where(labels == 'benign', np.random.exponential(scale=5000, size=n_samples), np.random.exponential(scale=150000, size=n_samples)),
            'total_entropy': np.where(labels == 'benign', np.random.normal(loc=15.0, scale=3.0, size=n_samples).clip(0), np.random.normal(loc=35.0, scale=6.0, size=n_samples).clip(0)),
            'src_port': np.random.randint(1024, 65535, size=n_samples),
            'num_pkts_in': np.where(labels == 'benign', np.random.poisson(lam=12, size=n_samples), np.random.poisson(lam=180, size=n_samples)),
            'duration': np.where(labels == 'benign', np.random.exponential(scale=1.5, size=n_samples), np.random.exponential(scale=0.08, size=n_samples)),
            'avg_ipt': np.random.exponential(scale=15.0, size=n_samples),
            'entropy': np.where(labels == 'benign', np.random.uniform(2.0, 5.0, size=n_samples), np.random.uniform(5.5, 8.0, size=n_samples)),
            'bytes_in': np.where(labels == 'benign', np.random.exponential(scale=3000, size=n_samples), np.random.exponential(scale=50000, size=n_samples)),
            'num_pkts_out': np.where(labels == 'benign', np.random.poisson(lam=10, size=n_samples), np.random.poisson(lam=120, size=n_samples)),
            'proto': np.random.choice([6, 17], size=n_samples, p=[0.75, 0.25]),
            'label': labels
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(clean_dir, name), index=False)
        print(f"Generated {name}: {df.shape}")

    # Generate CIC-IDS2017 and CSE-CIC-IDS2018
    for ds_name in ['CIC-IDS2017.csv', 'CSE-CIC-IDS2018.csv']:
        labels = np.random.choice(['benign', 'malicious'], size=n_samples, p=[0.5, 0.5])
        # In 2018 concept drift alters distributions
        drift = 1.3 if '2018' in ds_name else 1.0
        data = {
            'Bwd Packet Length Std': np.where(labels == 'benign', np.random.exponential(20, size=n_samples), np.random.exponential(400 * drift, size=n_samples)),
            'Average Packet Size': np.where(labels == 'benign', np.random.normal(250, 50, size=n_samples).clip(20), np.random.normal(850 * drift, 200, size=n_samples).clip(40)),
            'Max Packet Length': np.where(labels == 'benign', np.random.uniform(100, 1500, size=n_samples), np.random.uniform(1400 * drift, 8000, size=n_samples)),
            'Packet Length Variance': np.where(labels == 'benign', np.random.exponential(500, size=n_samples), np.random.exponential(25000 * drift, size=n_samples)),
            'Packet Length Std': np.where(labels == 'benign', np.random.uniform(5, 50, size=n_samples), np.random.uniform(80 * drift, 400, size=n_samples)),
            'Avg Bwd Segment Size': np.where(labels == 'benign', np.random.uniform(10, 120, size=n_samples), np.random.uniform(200 * drift, 1200, size=n_samples)),
            'Packet Length Mean': np.where(labels == 'benign', np.random.uniform(50, 300, size=n_samples), np.random.uniform(300 * drift, 1000, size=n_samples)),
            'Destination Port': np.where(labels == 'benign', np.random.choice([80, 443, 53, 22], size=n_samples), np.random.choice([80, 21, 22, 8080, 3389], size=n_samples)),
            'Init_Win_bytes_forward': np.random.choice([8192, 29200, 65535, -1, 26883], size=n_samples),
            'Fwd Packet Length Mean': np.where(labels == 'benign', np.random.uniform(20, 150, size=n_samples), np.random.uniform(100 * drift, 600, size=n_samples)),
            'Init_Win_bytes_backward': np.random.choice([0, 29200, 65535, -1, 14600], size=n_samples),
            'Subflow Bwd Bytes': np.random.exponential(2000, size=n_samples),
            'Bwd Packet Length Max': np.random.uniform(50, 1500, size=n_samples),
            'Total Length of Bwd Packets': np.random.exponential(5000, size=n_samples),
            'Bwd Packet Length Mean': np.random.uniform(20, 400, size=n_samples),
            'Fwd Packet Length Max': np.random.uniform(50, 1500, size=n_samples),
            'Bwd Packet Length Min': np.random.uniform(0, 100, size=n_samples),
            'Subflow Fwd Bytes': np.random.exponential(2000, size=n_samples),
            'Avg Fwd Segment Size': np.random.uniform(20, 200, size=n_samples),
            'Fwd Header Length': np.random.choice([20, 32, 40, 60], size=n_samples),
            'label': labels
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(clean_dir, ds_name), index=False)
        print(f"Generated {ds_name}: {df.shape}")

if __name__ == '__main__':
    base = r"c:\Users\deore\Downloads\Evaluation-of-Machine-Learning-Algorithm-in-Network-Based-Intrusion-Detection-System-master\Evaluation-of-Machine-Learning-Algorithm-in-Network-Based-Intrusion-Detection-System-master"
    generate_benchmark_data(base)
    print("Benchmark data generated successfully!")
