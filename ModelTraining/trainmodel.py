import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "DataOps"))
from cleandata import FEATURE_COLS


def load_data_and_split():
    # Loads embedded dataset, separates features from labels, and splits into train/test sets
    data_path = Path(__file__).parent.parent / "Data" / "PostOpData" / "merged_embedded.parquet"
    df = pd.read_parquet(data_path)
    
    X = df.drop(columns=["gtin", "label_is_anomaly"], errors="ignore")
    y = df["label_is_anomaly"]
    
    print(f"Training on {len(X.columns)} features:")
    embedding_cols = [c for c in X.columns if '_emb_' in c]
    nutrient_cols = [c for c in X.columns if c in ['calories', 'total_fat', 'sat_fat', 'trans_fat', 'unsat_fat', 
                                                     'cholesterol', 'sodium', 'carbs', 'dietary_fiber',
                                                     'total_sugars', 'added_sugars', 'protein', 'potassium']]
    feature_cols = [c for c in X.columns if c in FEATURE_COLS]
    print(f"  - Text embeddings: {len(embedding_cols)} features")
    print(f"  - Nutrient columns: {len(nutrient_cols)} features")
    print(f"  - Engineered features: {len(feature_cols)} features")
    print(f"  - Total features: {len(X.columns)}")
    print(f"  - Training samples: {len(X):,}")
    print(f"  - Anomalies: {y.sum():,} ({y.mean()*100:.2f}%)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test


def train_smote_model(X_train, y_train):
    # Trains XGBoost model using SMOTE oversampling (creates synthetic samples by interpolating between existing minority class samples)
    print("\nTraining SMOTE model...")
    print("Applying SMOTE to training data...")
    
    smote = SMOTE(
        sampling_strategy=0.2,
        random_state=42,
        k_neighbors=5
    )
    
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE:")
    print(f"  - Training samples: {len(X_train_smote):,}")
    print(f"  - Anomalies: {y_train_smote.sum():,} ({y_train_smote.mean()*100:.2f}%)")
    
    models_dir = Path(__file__).parent.parent / "models"
    with open(models_dir / "best_hyperparams.json", "r") as f:
        best_params = json.load(f)
    
    # scale_pos_weight=1.0 because SMOTE already handles class imbalance
    model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=1.0,
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=42
    )
    
    model.fit(X_train_smote, y_train_smote)
    
    model_path = models_dir / "bitecheckSMOTE_V1.json"
    model.get_booster().save_model(str(model_path))
    print(f"SMOTE model saved as {model_path}")
    
    return model


def train_adasyn_model(X_train, y_train):
    # Trains XGBoost model using ADASYN oversampling (focuses on harder-to-learn samples near class boundaries)
    print("\nTraining ADASYN model...")
    print("Applying ADASYN to training data...")
    
    adasyn = ADASYN(
        sampling_strategy=0.2,
        random_state=42,
        n_neighbors=5
    )
    
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
    
    print(f"After ADASYN:")
    print(f"  - Training samples: {len(X_train_adasyn):,}")
    print(f"  - Anomalies: {y_train_adasyn.sum():,} ({y_train_adasyn.mean()*100:.2f}%)")
    
    models_dir = Path(__file__).parent.parent / "models"
    with open(models_dir / "best_hyperparams.json", "r") as f:
        best_params = json.load(f)
    
    # scale_pos_weight=1.0 because ADASYN already handles class imbalance
    model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=1.0,
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=42
    )
    
    model.fit(X_train_adasyn, y_train_adasyn)
    
    model_path = models_dir / "bitecheckADASYN_V1.json"
    model.get_booster().save_model(str(model_path))
    print(f"ADASYN model saved as {model_path}")
    
    return model


def evaluate_ensemble(model_smote, model_adasyn, X_test, y_test):
    # Evaluates ensemble performance using max probability strategy (flags item if EITHER model flags it for higher recall)
    print("\nEvaluating ensemble (max probability)...")
    
    probs_smote = model_smote.predict_proba(X_test)[:, 1]
    probs_adasyn = model_adasyn.predict_proba(X_test)[:, 1]
    probs_ensemble = np.maximum(probs_smote, probs_adasyn)
    
    print("\nEnsemble Sensitivity Analysis (Max Probability):")
    thresholds = [0.7, 0.5, 0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2]
    for t in thresholds:
        t_preds = (probs_ensemble >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, t_preds).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_test, t_preds)
        print(f"Threshold {t:.2f}: Recall = {recall:.1%}, Precision = {precision:.1%}, F1 = {f1:.3f}, Total Flagged = {tp+fp}")
    
    print("\nIndividual Model Comparison:")
    print("SMOTE model performance:")
    for t in [0.2, 0.25, 0.3]:
        t_preds = (probs_smote >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, t_preds).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"  Threshold {t:.2f}: Recall = {recall:.1%}, Precision = {precision:.1%}")
    
    print("ADASYN model performance:")
    for t in [0.2, 0.25, 0.3]:
        t_preds = (probs_adasyn >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, t_preds).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"  Threshold {t:.2f}: Recall = {recall:.1%}, Precision = {precision:.1%}")


def train_final_model():
    # Main function: trains both SMOTE and ADASYN models, then evaluates ensemble performance
    X_train, X_test, y_train, y_test = load_data_and_split()
    
    model_smote = train_smote_model(X_train, y_train)
    model_adasyn = train_adasyn_model(X_train, y_train)
    
    evaluate_ensemble(model_smote, model_adasyn, X_test, y_test)
    
    models_dir = Path(__file__).parent.parent / "models"
    print(f"\nBoth models saved successfully!")
    print(f"  - SMOTE: {models_dir / 'bitecheckSMOTE_V1.json'}")
    print(f"  - ADASYN: {models_dir / 'bitecheckADASYN_V1.json'}")


if __name__ == "__main__":
    train_final_model()
