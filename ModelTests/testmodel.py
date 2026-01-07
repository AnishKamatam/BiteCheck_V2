import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
sys.path.append(str(Path(__file__).parent.parent / "DataOps"))
from cleandata import FEATURE_COLS, add_nutrition_features

TEXT_COLS = ["category_name", "product_name", "ingredients_text"]

NUTRIENT_COLS = [
    "calories", "total_fat", "sat_fat", "trans_fat", "unsat_fat",
    "cholesterol", "sodium", "carbs", "dietary_fiber",
    "total_sugars", "added_sugars", "protein", "potassium",
]

COLS_TO_KEEP = TEXT_COLS + NUTRIENT_COLS + FEATURE_COLS + ["gtin"]


def process_and_embed(df):
    # Cleans and embeds raw CSV data to match training feature set
    keep = [c for c in COLS_TO_KEEP if c in df.columns]
    df = df[keep].copy()
    print(f"Selected {len(keep)} columns: text + nutrients + engineered features + gtin")
    
    # Convert text columns to string type
    for col in ["gtin"] + TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    # Clean nutrient columns: remove carets and convert to numeric
    for col in NUTRIENT_COLS:
        if col in df.columns:
            numeric_val = df[col].astype(str).str.rstrip("^").str.strip()
            df[col] = pd.to_numeric(numeric_val, errors="coerce")
            df[col] = df[col].fillna(-1)
    
    # Remove rows without GTIN or empty ingredients
    df = df.dropna(subset=["gtin"])
    if "ingredients_text" in df.columns:
        df = df[df["ingredients_text"].notna() & (df["ingredients_text"].astype(str).str.strip() != "")]
    
    # Generate engineered features for anomaly detection
    df = add_nutrition_features(df)
    
    # Fill missing engineered features with 0
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Embed text columns using SentenceTransformer
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for col in TEXT_COLS:
        if col in df.columns:
            print(f"Embedding {col}...")
            text_data = df[col].fillna("").astype(str)
            embeddings = model.encode(text_data.tolist())
            embedding_cols = [f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
            embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
            df = pd.concat([df, embedding_df], axis=1)
    
    # Remove original text columns after embedding
    df = df.drop(columns=TEXT_COLS, errors="ignore")
    
    final_cols = df.columns.tolist()
    embedding_count = len([c for c in final_cols if '_emb_' in c])
    nutrient_count = len([c for c in final_cols if c in NUTRIENT_COLS])
    feature_count = len([c for c in final_cols if c in FEATURE_COLS])
    print(f"Final features: {embedding_count} embeddings + {nutrient_count} nutrients + {feature_count} engineered = {len(final_cols)} total")
    
    gtin_column = df["gtin"].copy() if "gtin" in df.columns else None
    X = df.drop(columns=["gtin"], errors="ignore")
    
    return X, gtin_column


def test_model_on_dataset(csv_path, expected_result, dataset_name, threshold=0.3):
    # Tests model on a dataset using custom probability threshold
    print(f"\nTesting on: {dataset_name}")
    print(f"Expected: {expected_result} | Using Threshold: {threshold}")
    
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return 0, None, None
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows")
    
    # Extract sodium values for hard rule checking
    sodium_values = None
    if "sodium" in df.columns:
        sodium_series = df["sodium"].astype(str).str.rstrip("^").str.strip()
        sodium_values = pd.to_numeric(sodium_series, errors="coerce")
        has_gtin = df["gtin"].notna() if "gtin" in df.columns else pd.Series([True] * len(df))
        has_ingredients = df["ingredients_text"].notna() & (df["ingredients_text"].astype(str).str.strip() != "") if "ingredients_text" in df.columns else pd.Series([True] * len(df))
        sodium_values = sodium_values[has_gtin & has_ingredients].reset_index(drop=True)
    
    # Process and embed data to match training format
    X, gtin_column = process_and_embed(df)
    print(f"After processing: {len(X):,} rows, {len(X.columns)} features")
    
    # Load ensemble models
    models_dir = Path(__file__).parent.parent / "models"
    smote_path = models_dir / "bitecheckSMOTE_V1.json"
    adasyn_path = models_dir / "bitecheckADASYN_V1.json"
    
    if not smote_path.exists() or not adasyn_path.exists():
        print(f"Error: Ensemble models not found")
        print(f"  Looking for: {smote_path}")
        print(f"  Looking for: {adasyn_path}")
        return 0, None, None
    
    print(f"Loading ensemble models...")
    print(f"  - SMOTE model: {smote_path.name}")
    print(f"  - ADASYN model: {adasyn_path.name}")
    
    model_smote = xgb.XGBClassifier()
    model_smote.load_model(str(smote_path))
    
    model_adasyn = xgb.XGBClassifier()
    model_adasyn.load_model(str(adasyn_path))
    
    # Reorder features to match model's expected order
    expected_features = model_smote.get_booster().feature_names
    print(f"Model expects {len(expected_features)} features in specific order")
    
    missing_features = [f for f in expected_features if f not in X.columns]
    if missing_features:
        print(f"Warning: Missing {len(missing_features)} features in test data")
        print(f"  Missing: {missing_features[:10]}...")
    
    X_reordered = X[[f for f in expected_features if f in X.columns]].copy()
    
    # Add missing features with zero values
    for f in missing_features:
        X_reordered[f] = 0
    
    X_reordered = X_reordered[expected_features]
    
    # Generate predictions from both models
    print("Generating predictions from both models...")
    probs_smote = model_smote.predict_proba(X_reordered)[:, 1]
    probs_adasyn = model_adasyn.predict_proba(X_reordered)[:, 1]
    
    # Use max probability for ensemble (aggressive recall)
    probs = np.maximum(probs_smote, probs_adasyn)
    print(f"Using ensemble: Max probability (aggressive recall)")
    
    # Apply threshold to get binary predictions
    preds = (probs >= threshold).astype(int)
    
    # Apply sodium hard rule (>1500mg)
    if sodium_values is not None and len(sodium_values) == len(preds):
        high_sodium_mask = sodium_values > 1500
        high_sodium_count = high_sodium_mask.sum()
        if high_sodium_count > 0:
            preds[high_sodium_mask] = 1
            print(f"\nRule Applied: {high_sodium_count} items flagged due to sodium > 1500mg")
    elif "sodium" in X.columns:
        sodium_from_X = X["sodium"] if "sodium" in X.columns else None
        if sodium_from_X is not None:
            high_sodium_mask = sodium_from_X > 1500
            high_sodium_count = high_sodium_mask.sum()
            if high_sodium_count > 0:
                preds[high_sodium_mask] = 1
                print(f"\nRule Applied: {high_sodium_count} items flagged due to sodium > 1500mg")
    
    # Apply deterministic hard rules for logical inconsistencies
    rule_flags = np.zeros(len(preds), dtype=int)
    total_rules_flagged = 0
    
    # Hard Rule 1: Added sugars cannot exceed total sugars
    if "added_sugars" in X.columns and "total_sugars" in X.columns:
        mask = X["added_sugars"] > (X["total_sugars"] + 0.5)
        count = mask.sum()
        if count > 0:
            rule_flags[mask] = 1
            total_rules_flagged += count
            print(f"Hard Rule 1: {count} items flagged (added_sugars > total_sugars + 0.5)")
    
    # Hard Rule 2: Total sugars cannot exceed carbs
    if "total_sugars" in X.columns and "carbs" in X.columns:
        mask = X["total_sugars"] > (X["carbs"] + 1.0)
        count = mask.sum()
        if count > 0:
            rule_flags[mask] = 1
            total_rules_flagged += count
            print(f"Hard Rule 2: {count} items flagged (total_sugars > carbs + 1.0)")
    
    # Hard Rule 3: Dietary fiber cannot exceed carbs
    if "dietary_fiber" in X.columns and "carbs" in X.columns:
        mask = X["dietary_fiber"] > (X["carbs"] + 1.0)
        count = mask.sum()
        if count > 0:
            rule_flags[mask] = 1
            total_rules_flagged += count
            print(f"Hard Rule 3: {count} items flagged (dietary_fiber > carbs + 1.0)")
    
    # Hard Rule 4: Saturated fat cannot exceed total fat
    if "sat_fat" in X.columns and "total_fat" in X.columns:
        mask = X["sat_fat"] > (X["total_fat"] + 0.5)
        count = mask.sum()
        if count > 0:
            rule_flags[mask] = 1
            total_rules_flagged += count
            print(f"Hard Rule 4: {count} items flagged (sat_fat > total_fat + 0.5)")
    
    # Hard Rule 5: Trans fat cannot exceed total fat
    if "trans_fat" in X.columns and "total_fat" in X.columns:
        mask = X["trans_fat"] > (X["total_fat"] + 0.5)
        count = mask.sum()
        if count > 0:
            rule_flags[mask] = 1
            total_rules_flagged += count
            print(f"Hard Rule 5: {count} items flagged (trans_fat > total_fat + 0.5)")
    
    # Hard Rule 6: Unsaturated fat cannot exceed total fat
    if "unsat_fat" in X.columns and "total_fat" in X.columns:
        mask = X["unsat_fat"] > (X["total_fat"] + 0.5)
        count = mask.sum()
        if count > 0:
            rule_flags[mask] = 1
            total_rules_flagged += count
            print(f"Hard Rule 6: {count} items flagged (unsat_fat > total_fat + 0.5)")
    
    # Hard Rule 7: Sum of fat subtypes cannot exceed total fat
    if all(col in X.columns for col in ["sat_fat", "trans_fat", "unsat_fat", "total_fat"]):
        fat_sum = X["sat_fat"] + X["trans_fat"] + X["unsat_fat"]
        mask = fat_sum > (X["total_fat"] + 1.0)
        count = mask.sum()
        if count > 0:
            rule_flags[mask] = 1
            total_rules_flagged += count
            print(f"Hard Rule 7: {count} items flagged (fat_sum > total_fat + 1.0)")
    
    # Apply hard rule flags to predictions
    preds[rule_flags == 1] = 1
    if total_rules_flagged > 0:
        print(f"\nTotal items auto-flagged by hard rules: {total_rules_flagged}")
    
    total_rows = len(preds)
    anomalies_detected = sum(preds)
    anomaly_percentage = (anomalies_detected / total_rows * 100) if total_rows > 0 else 0
    avg_probability = probs.mean()
    
    print(f"\nResults (at {threshold} threshold):")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Anomalies detected: {anomalies_detected:,}")
    print(f"  Catch Rate / FP Rate: {anomaly_percentage:.2f}%")
    print(f"  Average probability: {avg_probability:.3f}")
    
    high = sum(probs >= 0.7)
    med = sum((probs >= 0.3) & (probs < 0.7))
    low = sum(probs < 0.3)
    
    print(f"\nConfidence Breakdown:")
    print(f"  High (>=0.7): {high} | Medium (0.3-0.7): {med} | Low (<0.3): {low}")
    
    return anomaly_percentage, probs, preds


def main():
    # Tests model on both validation datasets and prints performance summary
    data_folder = Path(__file__).parent.parent / "TestData"
    
    TARGET_THRESHOLD = 0.2 
    
    errors_file = data_folder / "AI Training Data - items data errors.csv"
    catch_rate, _, _ = test_model_on_dataset(errors_file, "100% anomalies", "Items Data Errors", threshold=TARGET_THRESHOLD)
    
    approved_file = data_folder / "AI Training Data - approved profiles.csv"
    fp_rate, _, _ = test_model_on_dataset(approved_file, "0% anomalies", "Approved Profiles", threshold=TARGET_THRESHOLD)
    
    print(f"\nFINAL PERFORMANCE SUMMARY")
    print(f"Overall Catch Rate (Recall): {catch_rate:.2f}%")
    print(f"Overall False Positive Rate: {fp_rate:.2f}%")
    
    if fp_rate <= 10.0:
        print("SUCCESS: Threshold met the <10% False Positive requirement!")
    else:
        print(f"ALERT: False Positive rate exceeds 10%. Consider raising threshold above {TARGET_THRESHOLD}.")


if __name__ == "__main__":
    main()
