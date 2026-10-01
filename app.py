import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Set Streamlit page config
st.set_page_config(
    page_title="N-IDS ML Evaluation & Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #93c5fd;
        margin: 8px 0 0 0;
        font-size: 1.05rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .threat-banner-malicious {
        background-color: #fee2e2;
        border-left: 6px solid #ef4444;
        color: #991b1b;
        padding: 16px;
        border-radius: 6px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 16px;
    }
    .threat-banner-benign {
        background-color: #dcfce7;
        border-left: 6px solid #22c55e;
        color: #166534;
        padding: 16px;
        border-radius: 6px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# App Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Published Paper Empirical Results
PAPER_RESULTS = {
    'CIC': {
        'In_Dist_Name': 'CIC-IDS2017 (In-Distribution)',
        'Prog_Name': 'CSE-CIC-IDS2018 (Progressive / Drift)',
        'Accuracy': {
            'Decision Tree': {'train': 0.9959, 'test': 0.5942},
            'Random Forest': {'train': 0.9964, 'test': 0.5949},
            'Support Vector Machine': {'train': 0.9679, 'test': 0.7559},
            'Naive Bayes': {'train': 0.8705, 'test': 0.4972},
            'Artificial Neural Network': {'train': 0.9549, 'test': 0.7000},
            'Deep Neural Network': {'train': 0.9735, 'test': 0.6518},
        },
        'F1': {
            'Decision Tree': {'train': 0.9959, 'test': 0.5286},
            'Random Forest': {'train': 0.9964, 'test': 0.5294},
            'Support Vector Machine': {'train': 0.9679, 'test': 0.7503},
            'Naive Bayes': {'train': 0.8697, 'test': 0.3331},
            'Artificial Neural Network': {'train': 0.9549, 'test': 0.6869},
            'Deep Neural Network': {'train': 0.9735, 'test': 0.6216},
        },
        'Train_Time': {
            'Decision Tree': 1.2,
            'Random Forest': 22.4,
            'Support Vector Machine': 148.5,
            'Naive Bayes': 0.4,
            'Artificial Neural Network': 34.2,
            'Deep Neural Network': 41.8,
        }
    },
    'LUFlow': {
        'In_Dist_Name': 'LUFlow 2020 (In-Distribution)',
        'Prog_Name': 'LUFlow 2021 (Progressive / Drift)',
        'Accuracy': {
            'Decision Tree': {'train': 0.9994, 'test': 0.9990},
            'Random Forest': {'train': 0.9994, 'test': 0.9991},
            'Support Vector Machine': {'train': 0.9956, 'test': 0.9934},
            'Naive Bayes': {'train': 0.7276, 'test': 0.7377},
            'Artificial Neural Network': {'train': 0.9960, 'test': 0.9930},
            'Deep Neural Network': {'train': 0.9982, 'test': 0.9975},
        },
        'F1': {
            'Decision Tree': {'train': 0.9994, 'test': 0.9990},
            'Random Forest': {'train': 0.9994, 'test': 0.9991},
            'Support Vector Machine': {'train': 0.9956, 'test': 0.9934},
            'Naive Bayes': {'train': 0.7087, 'test': 0.7226},
            'Artificial Neural Network': {'train': 0.9960, 'test': 0.9930},
            'Deep Neural Network': {'train': 0.9982, 'test': 0.9975},
        },
        'Train_Time': {
            'Decision Tree': 1.8,
            'Random Forest': 38.6,
            'Support Vector Machine': 195.2,
            'Naive Bayes': 0.6,
            'Artificial Neural Network': 45.1,
            'Deep Neural Network': 56.4,
        }
    }
}

@st.cache_resource
def load_and_train_baseline_models(dataset_choice='LUFlow'):
    """Load benchmark data and train baseline models for real-time inference."""
    if dataset_choice == 'LUFlow':
        train_file = os.path.join(BASE_DIR, 'Dataset', 'dataset_cleaned', 'LUFlow.csv')
        test_file = os.path.join(BASE_DIR, 'Dataset', 'dataset_cleaned', 'LUFlow2021.csv')
        fs_file = os.path.join(BASE_DIR, 'Dataset', 'features_selected', 'LUFlow_RandomForestClassifier_11.csv')
        n_feat = 6
    else:
        train_file = os.path.join(BASE_DIR, 'Dataset', 'dataset_cleaned', 'CIC-IDS2017.csv')
        test_file = os.path.join(BASE_DIR, 'Dataset', 'dataset_cleaned', 'CSE-CIC-IDS2018.csv')
        fs_file = os.path.join(BASE_DIR, 'Dataset', 'features_selected', 'CIC-IDS2017_RandomForestClassifier_20.csv')
        n_feat = 11

    if not os.path.exists(train_file):
        from generate_benchmark_data import generate_benchmark_data
        generate_benchmark_data(BASE_DIR)

    features = pd.read_csv(fs_file, header=None).squeeze()
    if isinstance(features, pd.DataFrame):
        features = features.iloc[:, 0]
    selected_features = list(features[:n_feat])

    cols = selected_features + ['label']
    df_train = pd.read_csv(train_file, usecols=cols)
    df_test = pd.read_csv(test_file, usecols=cols)

    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Decision Tree': tree.DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.000138, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Support Vector Machine': SVC(C=10, kernel='rbf', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(var_smoothing=8.11e-06),
        'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(40,), max_iter=250, random_state=42)
    }

    trained = {}
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        trained[name] = clf

    return {
        'models': trained,
        'scaler': scaler,
        'features': selected_features,
        'df_train': df_train,
        'df_test': df_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test
    }

# ----------------- UI HEADER -----------------
st.markdown("""
<div class="main-header">
    <h1>🛡️ Evaluation of Machine Learning in Network Intrusion Detection</h1>
    <p>A Progressive Dataset Benchmark & Concept Drift Analysis System (MDPI Symmetry 2023)</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.shields.io/badge/Status-Active-brightgreen", use_container_width=False)
    st.markdown("### 📌 Project Configuration")
    dataset_selection = st.selectbox(
        "Active IDS Framework",
        ["LUFlow (2020 vs 2021)", "CIC-IDS (2017 vs 2018)"],
        index=0,
        help="Switch between LUFlow progressive flows or CIC-IDS flows."
    )
    current_ds_key = 'LUFlow' if 'LUFlow' in dataset_selection else 'CIC'
    
    st.divider()
    st.markdown("### ⚙️ Quick Actions")
    if st.button("🔄 Regenerate Sample Benchmark Data"):
        from generate_benchmark_data import generate_benchmark_data
        generate_benchmark_data(BASE_DIR)
        st.cache_resource.clear()
        st.success("Benchmark datasets regenerated!")

    st.markdown("### 📚 Reference Thesis")
    st.caption("**Paper:** *Evaluation of ML Algorithms in Network-Based Intrusion Detection Using Progressive Dataset*")
    st.caption("**Journal:** MDPI Symmetry 2023, 15(6), 1251")
    st.caption("**Models:** DT, RF, SVM, Naive Bayes, ANN, DNN")

# Load baseline models
engine_data = load_and_train_baseline_models(current_ds_key)

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Empirical Findings & Concept Drift",
    "🚀 Live Pipeline & Model Training",
    "🛡️ Real-Time Threat Simulator",
    "🔍 Feature Importance & Selection",
    "📁 Batch Flow CSV Inspector"
])

# ================= TAB 1: EMPIRICAL FINDINGS =================
with tab1:
    st.markdown("### 🔬 Published Research: Progressive Datasets & Concept Drift")
    st.write(
        "A central problem in Network Intrusion Detection Systems (N-IDS) is **Concept Drift**: "
        "attack vectors and normal network behavior evolve over time. "
        "While models achieve **>99% accuracy** on the same dataset they were trained on, their performance often collapses "
        "when deployed against future traffic."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Key Finding 1 (CIC-IDS2017 → CSE-CIC-IDS2018):**\n\nDecision Tree and Random Forest experienced severe accuracy degradation (dropping from **~99.6% down to ~59.4%**) due to overfitting to ephemeral port and packet lengths. SVM proved the most resilient classical model (**75.6%**).")
    with col2:
        st.success("💡 **Key Finding 2 (LUFlow 2020 → LUFlow 2021):**\n\nBy incorporating entropy and bi-directional flow features rather than brittle static packet signatures, all models retained exceptional generalization (**>99.3% accuracy**), confirming LUFlow's temporal robustness.")

    col_chart1, col_chart2 = st.columns(2)

    # CIC Performance Comparison
    with col_chart1:
        st.subheader("CIC-IDS: 2017 vs 2018 (High Concept Drift)")
        cic_data = PAPER_RESULTS['CIC']['Accuracy']
        models_list = list(cic_data.keys())
        train_acc = [cic_data[m]['train'] * 100 for m in models_list]
        test_acc = [cic_data[m]['test'] * 100 for m in models_list]

        fig_cic = go.Figure(data=[
            go.Bar(name='CIC-IDS2017 (In-Dist)', x=models_list, y=train_acc, marker_color='#3b82f6'),
            go.Bar(name='CSE-CIC-IDS2018 (1-Year Drift)', x=models_list, y=test_acc, marker_color='#ef4444')
        ])
        fig_cic.update_layout(
            barmode='group',
            yaxis=dict(title='Accuracy (%)', range=[0, 105]),
            xaxis_tickangle=-25,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cic, use_container_width=True)

    # LUFlow Performance Comparison
    with col_chart2:
        st.subheader("LUFlow: July 2020 vs Jan 2021 (Resilient)")
        lu_data = PAPER_RESULTS['LUFlow']['Accuracy']
        lu_models = list(lu_data.keys())
        lu_train_acc = [lu_data[m]['train'] * 100 for m in lu_models]
        lu_test_acc = [lu_data[m]['test'] * 100 for m in lu_models]

        fig_lu = go.Figure(data=[
            go.Bar(name='LUFlow 2020 (In-Dist)', x=lu_models, y=lu_train_acc, marker_color='#10b981'),
            go.Bar(name='LUFlow 2021 (6-Month Drift)', x=lu_models, y=lu_test_acc, marker_color='#6366f1')
        ])
        fig_lu.update_layout(
            barmode='group',
            yaxis=dict(title='Accuracy (%)', range=[0, 105]),
            xaxis_tickangle=-25,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_lu, use_container_width=True)

    st.subheader("⏱️ Training Time Consumption (CPU Seconds)")
    tt_cic = PAPER_RESULTS['CIC']['Train_Time']
    tt_lu = PAPER_RESULTS['LUFlow']['Train_Time']
    fig_time = go.Figure(data=[
        go.Bar(name='CIC Training Time (s)', x=list(tt_cic.keys()), y=list(tt_cic.values()), marker_color='#f59e0b'),
        go.Bar(name='LUFlow Training Time (s)', x=list(tt_lu.keys()), y=list(tt_lu.values()), marker_color='#06b6d4')
    ])
    fig_time.update_layout(
        barmode='group',
        yaxis=dict(title='Seconds (log scale)', type='log'),
        xaxis_tickangle=-20,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_time, use_container_width=True)

# ================= TAB 2: LIVE MODEL TRAINER =================
with tab2:
    st.markdown("### 🚀 Execute Machine Learning Pipeline Live")
    st.write("Train and evaluate the selected models on the active dataset in real time:")

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.markdown("#### Training Parameters")
        selected_model_names = st.multiselect(
            "Select Algorithms to Train",
            ['Decision Tree', 'Random Forest', 'Support Vector Machine', 'Naive Bayes', 'Artificial Neural Network', 'Deep Neural Network'],
            default=['Decision Tree', 'Random Forest', 'Support Vector Machine', 'Naive Bayes']
        )
        rf_trees = st.slider("Random Forest Trees (n_estimators)", 10, 200, 50, 10)
        test_split_ratio = st.slider("Validation Split Ratio", 0.1, 0.4, 0.2, 0.05)
        run_btn = st.button("⚡ Run Training & Validation Pipeline", type="primary")

    with col_t2:
        if run_btn:
            if not selected_model_names:
                st.warning("Please select at least one algorithm.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Loading and scaling features...")
                time.sleep(0.2)
                progress_bar.progress(20)

                df_tr = engine_data['df_train']
                df_te = engine_data['df_test']
                feat_cols = engine_data['features']

                X = df_tr[feat_cols]
                y = df_tr['label']
                X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=test_split_ratio, random_state=42, stratify=y)

                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_va_sc = scaler.transform(X_va)
                X_te_sc = scaler.transform(df_te[feat_cols])
                y_te = df_te['label']

                all_model_defs = {
                    'Decision Tree': tree.DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.000138),
                    'Random Forest': RandomForestClassifier(n_estimators=rf_trees, max_depth=10, random_state=42, n_jobs=-1),
                    'Support Vector Machine': SVC(C=10, kernel='rbf', probability=True, random_state=42),
                    'Naive Bayes': GaussianNB(var_smoothing=8.11e-06),
                    'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(40,), max_iter=250, random_state=42),
                    'Deep Neural Network': MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=250, random_state=42)
                }

                live_metrics = []
                step_increment = 70 // len(selected_model_names)
                current_prog = 20

                for name in selected_model_names:
                    status_text.text(f"Training {name}...")
                    clf = all_model_defs[name]
                    t_start = time.process_time()
                    clf.fit(X_tr_sc, y_tr)
                    t_elapsed = time.process_time() - t_start

                    # In-distribution
                    pred_va = clf.predict(X_va_sc)
                    acc_va = metrics.accuracy_score(y_va, pred_va)
                    f1_va = metrics.f1_score(y_va, pred_va, pos_label='malicious', zero_division=0)
                    prec_va = metrics.precision_score(y_va, pred_va, pos_label='malicious', zero_division=0)
                    rec_va = metrics.recall_score(y_va, pred_va, pos_label='malicious', zero_division=0)

                    # Progressive out-of-distribution
                    pred_te = clf.predict(X_te_sc)
                    acc_te = metrics.accuracy_score(y_te, pred_te)
                    f1_te = metrics.f1_score(y_te, pred_te, pos_label='malicious', zero_division=0)

                    live_metrics.append({
                        'Algorithm': name,
                        'In-Dist Accuracy': f"{acc_va*100:.2f}%",
                        'In-Dist Precision': f"{prec_va*100:.2f}%",
                        'In-Dist Recall': f"{rec_va*100:.2f}%",
                        'In-Dist F1-Score': f"{f1_va:.4f}",
                        'Progressive Test Acc': f"{acc_te*100:.2f}%",
                        'Progressive F1-Score': f"{f1_te:.4f}",
                        'Train Time (s)': f"{t_elapsed:.3f}s"
                    })

                    current_prog += step_increment
                    progress_bar.progress(min(current_prog, 95))

                progress_bar.progress(100)
                status_text.text("Training & Evaluation Completed!")
                st.success("✅ Models trained and benchmarked successfully!")

                st.dataframe(pd.DataFrame(live_metrics), use_container_width=True)
        else:
            st.info("Select algorithms and click **Run Training & Validation Pipeline** above to begin.")

# ================= TAB 3: REAL-TIME THREAT SIMULATOR =================
with tab3:
    st.markdown("### 🛡️ Live Network Packet / Flow Threat Analyzer")
    st.write("Inspect incoming network traffic flows in real-time. Choose a pre-configured attack vector or adjust flow attributes:")

    st.markdown("#### ⚡ Quick Attack Presets")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    
    preset_chosen = None
    if col_p1.button("💥 DDoS Flood Attack"):
        preset_chosen = "ddos"
    if col_p2.button("🔍 Port Scanning Recon"):
        preset_chosen = "portscan"
    if col_p3.button("🌐 Web Exploit / XSS / SQLi"):
        preset_chosen = "web"
    if col_p4.button("✅ Normal Benign Traffic"):
        preset_chosen = "benign"

    st.markdown("#### Flow Attributes")
    feature_inputs = {}
    feats = engine_data['features']

    # Set default values according to preset or normal
    default_vals = {}
    if current_ds_key == 'LUFlow':
        if preset_chosen == "ddos":
            default_vals = {'dest_port': 80, 'bytes_out': 250000, 'total_entropy': 42.0, 'src_port': 49152, 'num_pkts_in': 500, 'duration': 0.01}
        elif preset_chosen == "portscan":
            default_vals = {'dest_port': 1337, 'bytes_out': 120, 'total_entropy': 28.0, 'src_port': 58210, 'num_pkts_in': 2, 'duration': 0.05}
        elif preset_chosen == "web":
            default_vals = {'dest_port': 8080, 'bytes_out': 18000, 'total_entropy': 36.5, 'src_port': 51234, 'num_pkts_in': 45, 'duration': 0.8}
        else: # benign
            default_vals = {'dest_port': 443, 'bytes_out': 4200, 'total_entropy': 14.5, 'src_port': 52341, 'num_pkts_in': 15, 'duration': 1.2}
    else: # CIC
        if preset_chosen in ["ddos", "portscan", "web"]:
            default_vals = {
                'Bwd Packet Length Std': 450.0, 'Average Packet Size': 950.0, 'Max Packet Length': 3500.0,
                'Packet Length Variance': 18000.0, 'Packet Length Std': 150.0, 'Avg Bwd Segment Size': 600.0,
                'Packet Length Mean': 550.0, 'Destination Port': 80, 'Init_Win_bytes_forward': 29200,
                'Fwd Packet Length Mean': 280.0, 'Init_Win_bytes_backward': 0
            }
        else:
            default_vals = {
                'Bwd Packet Length Std': 15.0, 'Average Packet Size': 220.0, 'Max Packet Length': 1200.0,
                'Packet Length Variance': 250.0, 'Packet Length Std': 25.0, 'Avg Bwd Segment Size': 80.0,
                'Packet Length Mean': 180.0, 'Destination Port': 443, 'Init_Win_bytes_forward': 65535,
                'Fwd Packet Length Mean': 64.0, 'Init_Win_bytes_backward': 14600
            }

    # Render inputs in 3 columns
    cols_inputs = st.columns(3)
    for i, f in enumerate(feats):
        c = cols_inputs[i % 3]
        default = default_vals.get(f, 100.0)
        feature_inputs[f] = c.number_input(f"{f}", value=float(default), key=f"inp_{f}")

    if st.button("🚨 Analyze Network Flow", type="primary", use_container_width=True):
        input_df = pd.DataFrame([feature_inputs])
        input_scaled = engine_data['scaler'].transform(input_df)

        model_preds = {}
        model_probs = {}
        for m_name, model in engine_data['models'].items():
            pred = model.predict(input_scaled)[0]
            model_preds[m_name] = pred
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_scaled)[0]
                classes = list(model.classes_)
                mal_idx = classes.index('malicious') if 'malicious' in classes else 1
                model_probs[m_name] = probs[mal_idx]
            else:
                model_probs[m_name] = 1.0 if pred == 'malicious' else 0.0

        # Consensus
        mal_votes = sum(1 for v in model_preds.values() if v == 'malicious')
        total_votes = len(model_preds)
        avg_mal_prob = np.mean(list(model_probs.values()))
        is_attack = avg_mal_prob >= 0.5

        st.divider()
        if is_attack:
            st.markdown(f"""
            <div class="threat-banner-malicious">
                🚨 SECURITY ALERT: MALICIOUS NETWORK ATTACK DETECTED
                <div style="font-size: 1rem; font-weight: normal; margin-top: 4px;">
                    Ensemble Confidence: {avg_mal_prob*100:.1f}% | {mal_votes}/{total_votes} Models Voted Threat
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="threat-banner-benign">
                ✅ FLOW VERIFIED: NORMAL / BENIGN NETWORK TRAFFIC
                <div style="font-size: 1rem; font-weight: normal; margin-top: 4px;">
                    Normal Traffic Confidence: {(1.0-avg_mal_prob)*100:.1f}% | {total_votes-mal_votes}/{total_votes} Models Voted Safe
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_mal_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Malicious Threat Probability (%)", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ef4444" if is_attack else "#22c55e"},
                    'steps': [
                        {'range': [0, 30], 'color': "#dcfce7"},
                        {'range': [30, 70], 'color': "#fef9c3"},
                        {'range': [70, 100], 'color': "#fee2e2"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_g2:
            st.markdown("#### Individual Model Decisions")
            df_decisions = pd.DataFrame([
                {
                    'Algorithm': name,
                    'Decision': 'MALICIOUS 🚨' if model_preds[name] == 'malicious' else 'BENIGN ✅',
                    'Malicious Probability': f"{model_probs[name]*100:.1f}%"
                }
                for name in model_preds
            ])
            st.dataframe(df_decisions, use_container_width=True)

# ================= TAB 4: FEATURE IMPORTANCE =================
with tab4:
    st.markdown("### 🔍 Random Forest Feature Importance Analysis")
    st.write(
        "Feature selection is critical to reducing computation time and removing redundant features. "
        "The project applies a two-stage method: **Random Forest Gini Importance** followed by **Brute Force forward subsetting**."
    )

    col_fi1, col_fi2 = st.columns(2)
    with col_fi1:
        st.subheader("LUFlow Dataset: Feature Importance")
        luflow_imp = {
            'dest_port': 0.347, 'bytes_out': 0.165, 'total_entropy': 0.126,
            'src_port': 0.106, 'num_pkts_in': 0.081, 'duration': 0.077,
            'avg_ipt': 0.026, 'entropy': 0.023, 'bytes_in': 0.021,
            'num_pkts_out': 0.020, 'proto': 0.008
        }
        df_luflow_imp = pd.DataFrame(list(luflow_imp.items()), columns=['Feature', 'Gini Score']).sort_values('Gini Score', ascending=True)
        fig_lu_fi = px.bar(df_luflow_imp, x='Gini Score', y='Feature', orientation='h', color='Gini Score', color_continuous_scale='teal')
        fig_lu_fi.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=400)
        st.plotly_chart(fig_lu_fi, use_container_width=True)
        st.caption("Top 6 selected features account for **>90%** of total discriminative information.")

    with col_fi2:
        st.subheader("CIC-IDS2017: Top Feature Importance")
        cic_imp = {
            'Bwd Packet Length Std': 0.075, 'Average Packet Size': 0.067, 'Max Packet Length': 0.047,
            'Packet Length Variance': 0.044, 'Packet Length Std': 0.043, 'Avg Bwd Segment Size': 0.038,
            'Packet Length Mean': 0.037, 'Destination Port': 0.037, 'Init_Win_bytes_forward': 0.031,
            'Fwd Packet Length Mean': 0.029, 'Init_Win_bytes_backward': 0.029
        }
        df_cic_imp = pd.DataFrame(list(cic_imp.items()), columns=['Feature', 'Gini Score']).sort_values('Gini Score', ascending=True)
        fig_cic_fi = px.bar(df_cic_imp, x='Gini Score', y='Feature', orientation='h', color='Gini Score', color_continuous_scale='plasma')
        fig_cic_fi.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=400)
        st.plotly_chart(fig_cic_fi, use_container_width=True)
        st.caption("Top 11 selected features encompass packet size distributions and TCP window sizes.")

# ================= TAB 5: BATCH FLOW CSV INSPECTOR =================
with tab5:
    st.markdown("### 📁 Batch Network Flow CSV Inspector")
    st.write("Scan entire network capture CSVs containing multiple flow records:")

    col_up1, col_up2 = st.columns([1, 1])
    with col_up1:
        uploaded_file = st.file_uploader("Upload Network Flow CSV", type=["csv"])
    with col_up2:
        st.markdown("Or run inspection on a benchmark test batch:")
        use_sample = st.button("📥 Load Sample Validation Batch (100 Flows)")

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif use_sample:
        batch_df = engine_data['df_test'].head(100).copy()

    if batch_df is not None:
        st.markdown(f"#### Processing {len(batch_df)} Flow Records")
        feat_cols = engine_data['features']
        
        # Verify required columns exist
        missing_cols = [c for c in feat_cols if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required flow columns in CSV: {missing_cols}")
        else:
            X_batch = batch_df[feat_cols]
            X_batch_sc = engine_data['scaler'].transform(X_batch)
            
            # Predict with Random Forest as primary engine
            rf_model = engine_data['models']['Random Forest']
            predictions = rf_model.predict(X_batch_sc)
            probs = rf_model.predict_proba(X_batch_sc)
            mal_prob = probs[:, list(rf_model.classes_).index('malicious')]

            result_df = batch_df.copy()
            result_df['Classification'] = predictions
            result_df['Threat_Score_%'] = np.round(mal_prob * 100, 1)

            col_m1, col_m2, col_m3 = st.columns(3)
            num_attacks = sum(predictions == 'malicious')
            num_benign = len(predictions) - num_attacks
            col_m1.metric("Total Flows Analyzed", len(predictions))
            col_m2.metric("Malicious Attacks Flagged", num_attacks, delta=f"{(num_attacks/len(predictions))*100:.1f}%", delta_color="inverse")
            col_m3.metric("Normal / Benign Flows", num_benign)

            st.dataframe(result_df, use_container_width=True)

            csv_out = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Tagged Flow CSV",
                data=csv_out,
                file_name="classified_network_flows.csv",
                mime="text/csv"
            )

st.divider()
st.caption("Evaluation of Machine Learning Algorithm in Network-Based Intrusion Detection System | Built with Streamlit, Scikit-Learn & Plotly")
