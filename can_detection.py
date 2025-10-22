import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set random seed for reproducibility
RND = 42
np.random.seed(RND)

# 1) Generate synthetic CAN dataset
def generate_can_data(normal_n=2000, attack_n=600):
    """
    Creates a DataFrame with simulated normal and attack CAN messages.
    Columns: Timestamp, CAN_ID (int), DLC, Bytes (0..7), Label (0 normal, 1 attack)
    Attack types: flooding (high rate), spoofing (unexpected IDs), MASQUERADE (weird fixed payload), FUZZING (zeroed payload)
    """
    # Normal traffic: mix of typical CAN IDs
    normal_ids = [0x100, 0x200, 0x300, 0x400, 0x500]
    normal = pd.DataFrame({
        'Timestamp': np.cumsum(np.random.exponential(scale=0.01, size=normal_n)),
        'CAN_ID': np.random.choice(normal_ids, size=normal_n, p=[0.3,0.25,0.2,0.15,0.1]),
        'DLC': np.random.choice([8, 8, 8, 6, 7], size=normal_n),
    })
    # 8 data bytes
    for b in range(8):
        normal[f'D{b}'] = np.random.randint(0, 256, size=normal_n)
    normal['Label'] = 0

    # Attack traffic: mixture of flooding, spoofing, masquerade, fuzzing types

    # Flood: repeated same ID at high frequency
    flood_ids = [0x100]  # attacker floods an existing ID
    flood = pd.DataFrame({
        'Timestamp': np.cumsum(np.random.exponential(scale=0.001, size=int(attack_n*0.4))),
        'CAN_ID': np.random.choice(flood_ids, size=int(attack_n*0.4)),
        'DLC': 8,
    })
    for b in range(8):
        flood[f'D{b}'] = np.random.randint(0, 256, size=len(flood))

    # Spoof: new IDs not usually present
    spoof_ids = [0x900, 0x901, 0x902]
    spoof = pd.DataFrame({
        'Timestamp': np.cumsum(np.random.exponential(scale=0.005, size=int(attack_n*0.2))),
        'CAN_ID': np.random.choice(spoof_ids, size=int(attack_n*0.2)),
        'DLC': 8,
    })
    for b in range(8):
        spoof[f'D{b}'] = np.random.randint(0, 256, size=len(spoof))

    # Masquerade: use an existing ID with a weird, fixed payload
    masquerade_ids = [0x300] # An ID that is normally on the bus
    masquerade = pd.DataFrame({
        'Timestamp': np.cumsum(np.random.exponential(scale=0.008, size=int(attack_n*0.2))),
        'CAN_ID': np.random.choice(masquerade_ids, size=int(attack_n*0.2)),
        'DLC': 8,
    })
    # Payload is fixed and unusual, unlike normal random traffic for this ID
    for b in range(8):
        masquerade[f'D{b}'] = 0xFF # All bytes are 255

    # Fuzzing: use an existing ID with a zeroed-out payload
    fuzz_ids = [0x400] # Another ID that is normally on the bus
    fuzz = pd.DataFrame({
        'Timestamp': np.cumsum(np.random.exponential(scale=0.007, size=int(attack_n*0.2))),
        'CAN_ID': np.random.choice(fuzz_ids, size=int(attack_n*0.2)),
        'DLC': 8,
    })
    # Payload is all zeros, which might be unusual for this ID's normal operation
    for b in range(8):
        fuzz[f'D{b}'] = 0x00 # All bytes are 0
    
    attack = pd.concat([flood, spoof, masquerade, fuzz], ignore_index=True)
    attack['Label'] = 1

    # Merge and shuffle (important to mix timestamps correctly)
    df = pd.concat([normal, attack], ignore_index=True).sort_values('Timestamp').reset_index(drop=True)
    return df

# 2) Feature engineering
def feature_engineering(df):
    """
    Produces features that are informative for CAN IDS:
    - Delta time between messages (inter-arrival)
    - Rolling frequency counts per CAN_ID (window)
    - Byte-based simple stats (mean, entropy-ish) per message (we keep first byte and sum)
    - One-hot encode or map CAN_ID to integer (we keep raw ID as feature)
    """
    df = df.copy()
    # inter-arrival time
    df['Delta_t'] = df['Timestamp'].diff().fillna(df['Timestamp'].iloc[0])
    # frequency: count of same CAN_ID in last N messages (simple approach: global counts up to this row)
    df['ID_Count_Upto'] = df.groupby('CAN_ID').cumcount() + 1
    # map global frequency of ID (final count) as a feature (constant per ID)
    id_total_counts = df['CAN_ID'].value_counts()
    df['ID_Global_Count'] = df['CAN_ID'].map(id_total_counts)

    # payload-derived features: take byte 0 and sum of bytes
    byte_cols = [f'D{i}' for i in range(8)]
    df['Byte0'] = df['D0']  # could be specific important signal
    df['Payload_Sum'] = df[byte_cols].sum(axis=1)
    df['Payload_Mean'] = df[byte_cols].mean(axis=1)

    # Optionally: rolling features (e.g., messages per 0.1s window). We'll add a messages-per-0.1s feature:
    window = 0.1
    # count messages in [t - window, t)
    timestamps = df['Timestamp'].values
    counts = np.zeros(len(df), dtype=int)
    left = 0
    for i, t in enumerate(timestamps):
        while timestamps[left] < t - window:
            left += 1
        counts[i] = i - left + 1
    df['Msgs_in_0.1s'] = counts

    # Final feature set
    features = ['CAN_ID', 'DLC', 'Delta_t', 'ID_Count_Upto', 'ID_Global_Count',
                'Byte0', 'Payload_Sum', 'Payload_Mean', 'Msgs_in_0.1s']
    return df, features

# 3) Train supervised model (RandomForest) with evaluation
def train_evaluate_supervised(df, features):
    X = df[features]
    y = df['Label']

    # Train/test split (stratify to keep label ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=RND, stratify=y)

    # Simple RF baseline
    rf = RandomForestClassifier(n_estimators=150, random_state=RND, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:,1]

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    print("=== Confusion Matrix ===")
    print(cm)
    print("\n=== Classification Report ===")
    print(report)
    print(f"ROC AUC: {roc_auc:.4f}")

    return rf, (X_test, y_test, y_pred, y_proba, (fpr,tpr,roc_auc))

# 4) Unsupervised baseline: IsolationForest for anomaly detection
def unsupervised_isolation(df, features):
    X = df[features]
    # IsolationForest expects no label, we fit on full dataset or only normal subset in practice.
    iso = IsolationForest(contamination=0.1, random_state=RND)
    iso.fit(X)
    scores = -iso.decision_function(X)  # higher = more anomalous
    preds = iso.predict(X)  # -1 anomaly, 1 normal
    # Map to binary 1=attack, 0=normal for comparison with labels
    preds_binary = (preds == -1).astype(int)
    # Basic metrics
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(df['Label'], preds_binary)
    print(f"IsolationForest accuracy (naive compare): {acc:.4f}")
    return iso, preds_binary, scores

# 5) Utility: plotting
def plot_roc(fpr, tpr, roc_auc, out_dir):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
    plt.close()

def plot_confusion(cm, out_dir):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()

# 6) Main orchestration
def main():
    out_dir = 'results_can_ids'
    os.makedirs(out_dir, exist_ok=True)

    print("Generating data...")
    df = generate_can_data(normal_n=2000, attack_n=600)
    print("Total rows:", len(df), "Attack ratio:", df['Label'].mean())

    print("Engineering features...")
    df_feat, features = feature_engineering(df)

    # Optional: quick head
    print(df_feat[features + ['Label']].head())

    print("Training supervised model...")
    rf_model, eval_res = train_evaluate_supervised(df_feat, features)
    X_test, y_test, y_pred, y_proba, roc_pack = eval_res
    fpr, tpr, roc_auc = roc_pack

    # Save model
    model_path = os.path.join(out_dir, 'rf_can_ids_model.joblib')
    joblib.dump(rf_model, model_path)
    print("Saved RandomForest model to:", model_path)

    # Plot ROC and confusion
    plot_roc(fpr, tpr, roc_auc, out_dir)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, out_dir)

    # Unsupervised baseline
    print("Running IsolationForest (unsupervised) as baseline...")
    iso, preds_binary, scores = unsupervised_isolation(df_feat, features)
    joblib.dump(iso, os.path.join(out_dir, 'isolation_forest.joblib'))

    # Save dataset fragment for inspection
    df_feat.to_csv(os.path.join(out_dir, 'can_dataset_features.csv'), index=False)
    print("All results saved to:", out_dir)

if __name__ == '__main__':
    main()
