# BiteCheck: Anomaly Detection System for Nutrition Data Quality Assurance

A machine learning framework for automated detection of data quality anomalies in nutrition product datasets using ensemble gradient boosting and semantic text embeddings.

## Abstract

This repository implements an anomaly detection system designed to identify data entry errors and inconsistencies in structured nutrition product data. The system employs an ensemble of XGBoost classifiers trained on synthetically augmented datasets using SMOTE and ADASYN oversampling techniques. Text features are encoded using SentenceTransformer embeddings, enabling the model to capture semantic relationships in product descriptions, categories, and ingredient lists. The framework demonstrates effective performance on imbalanced classification tasks with approximately 6% positive class representation.

## System Architecture

```
BiteCheck_V2/
├── Data/
│   ├── PreOpData/          # Raw Excel files (Before/After pairs)
│   ├── PreOpDataCSV/       # Processed CSV files
│   └── PostOpData/         # Final embedded datasets
├── DataOps/
│   ├── addcaret.py         # Extract Excel data and add caret markers
│   ├── cleandata.py        # Data cleaning and feature engineering
│   ├── mergedata.py        # Merge Before/After datasets
│   └── embedtext.py        # Embed text columns using SentenceTransformer
├── ModelTraining/
│   ├── trainmodel.py       # Train ensemble models (SMOTE + ADASYN)
│   └── optimizemodel.py    # Hyperparameter optimization with Optuna
├── ModelTests/
│   └── testmodel.py        # Test ensemble model on validation datasets
└── models/
    ├── best_hyperparams.json           # Optimized hyperparameters
    ├── bitecheckSMOTE_V1.json         # SMOTE-trained model
    └── bitecheckADASYN_V1.json        # ADASYN-trained model
```

## Methodology

### Data Preprocessing Pipeline

#### 1. Data Extraction (`addcaret.py`)

The extraction module processes structured Excel files containing nutrition product data. The system identifies anomaly markers through visual indicators (yellow or cyan background colors) in the source data, converting these markers to caret (^) annotations for downstream processing.

#### 2. Data Cleaning (`cleandata.py`)

The cleaning module performs feature selection, type conversion, and data quality filtering. Anomaly labels are derived from caret markers present in the original dataset. Nutrient values are normalized to numeric format with missing values imputed using a sentinel value (-1). The pipeline removes records lacking essential identifiers (GTIN) or critical text fields (ingredients).

**Feature Engineering**: The cleaning module generates 27 engineered features capturing nutrition data consistency:

- **Calorie consistency** (3 features): Expected calories from macros, calorie gap, calorie gap ratio
- **Macro calorie shares** (4 features): Fat, carb, and protein calorie percentages, macro share sum
- **Sugar logic** (4 features): Added sugar ratios, sugar-to-carb ratios, violation flags
- **Fat structure** (3 features): Fat sum, fat gap, fat gap ratio
- **Nutrient density** (3 features): Sodium, sugar, and protein per calorie
- **Zero/missing flags** (10 features): Binary indicators for zero and missing values

#### 3. Data Merging (`mergedata.py`)

The merging module combines multiple data sources by aligning temporal snapshots (Before/After pairs) using product identifiers. Features are extracted from the "Before" state while labels are derived from the "After" state, creating a supervised learning dataset that captures the correction process.

#### 4. Text Embedding (`embedtext.py`)

Textual features are transformed into dense vector representations using the SentenceTransformer framework. The `all-MiniLM-L6-v2` model generates 384-dimensional embeddings for each text field (category name, product name, ingredients), resulting in 1,152 total embedding features that capture semantic relationships in product descriptions.

### Model Training Framework

#### Hyperparameter Optimization (`optimizemodel.py`)

Hyperparameter search is conducted using Optuna's tree-structured Parzen estimator (TPE) algorithm. The optimization targets Average Precision Score (PR-AUC), which is appropriate for imbalanced classification tasks. The search space includes tree structure parameters (max_depth, min_child_weight), regularization parameters (gamma, subsample, colsample_bytree), and learning rate. Optimal hyperparameters are persisted for model training.

#### Ensemble Model Training (`trainmodel.py`)

The training framework implements an ensemble approach using two complementary oversampling strategies:

1. **SMOTE Model**: Synthetic Minority Oversampling Technique generates synthetic samples through interpolation between existing minority class instances, creating uniform coverage across the feature space.

2. **ADASYN Model**: Adaptive Synthetic Sampling generates synthetic samples with higher density near class boundaries, focusing on harder-to-classify instances.

Both models utilize identical hyperparameter configurations and XGBoost architecture. Oversampling raises the minority class representation from approximately 6% to ~17% of the training set. Models are trained with `scale_pos_weight=1.0` since oversampling addresses class imbalance.

**Dataset Characteristics:**

- Feature dimensionality: 1,192 (1,152 text embeddings + 13 nutrient features + 27 engineered features)
- Training set size: 17,723 samples
- Initial class distribution: 6.09% positive class (1,079 anomalies)
- Post-SMOTE distribution: 15,978 samples, 16.67% positive class (2,663 anomalies)
- Post-ADASYN distribution: 15,966 samples, 16.60% positive class (2,651 anomalies)

#### Ensemble Strategy

The system employs a maximum probability ensemble strategy for inference. Predictions from both models are combined by taking the maximum probability score, effectively flagging instances when either model indicates high anomaly probability. This approach prioritizes recall while maintaining precision through complementary model perspectives.

**Ensemble Performance (at 0.20 threshold):**
- Recall: 53.2%
- Precision: 45.3%
- F1 Score: 0.489
- Total Flagged: 254 instances

### Model Evaluation

#### Test Dataset Performance

The system is evaluated on two validation datasets representing different data quality scenarios:

**Errors Dataset (Expected: 100% anomalies):**
- Detection rate (Recall): 88.33% (280 of 317 instances)
- Mean predicted probability: 0.667
- High confidence predictions (≥0.7): 184 instances
- Medium confidence predictions (0.3-0.7): 82 instances
- Low confidence predictions (<0.3): 51 instances
- Rule-based flags: 10 instances flagged due to sodium > 1500mg threshold
- Hard rule violations: 29 instances flagged by deterministic rules

**Approved Dataset (Expected: 0% anomalies):**
- False positive rate: 9.15% (29 of 317 instances)
- Mean predicted probability: 0.083
- High confidence predictions (≥0.7): 0 instances
- Medium confidence predictions (0.3-0.7): 8 instances
- Low confidence predictions (<0.3): 309 instances
- Rule-based flags: 9 instances flagged due to sodium > 1500mg threshold

The system achieves a false positive rate below the 10% operational requirement while maintaining high recall on known error cases.

## Feature Engineering

### Text Embeddings (1,152 features)

Semantic representations of textual product information:
- Category name: 384-dimensional embedding
- Product name: 384-dimensional embedding
- Ingredients text: 384-dimensional embedding

### Numerical Features (13 features)

Nutritional composition metrics:
`calories`, `total_fat`, `sat_fat`, `trans_fat`, `unsat_fat`, `cholesterol`, `sodium`, `carbs`, `dietary_fiber`, `total_sugars`, `added_sugars`, `protein`, `potassium`

### Engineered Features (27 features)

**Calorie Consistency:**
- `expected_calories`, `calorie_gap`, `calorie_gap_ratio`

**Macro Calorie Shares:**
- `fat_calorie_share`, `carb_calorie_share`, `protein_calorie_share`, `macro_share_sum`

**Sugar Logic:**
- `added_sugar_ratio`, `sugar_carb_ratio`, `added_sugars_gt_total`, `sugars_gt_carbs`

**Fat Structure:**
- `fat_sum`, `fat_gap`, `fat_gap_ratio`

**Nutrient Density:**
- `sodium_per_calorie`, `sugar_per_calorie`, `protein_per_calorie`

**Zero/Missing Flags:**
- `is_zero_calories`, `is_zero_total_fat`, `is_zero_carbs`, `is_zero_protein`, `is_zero_sodium`
- `is_missing_calories`, `is_missing_total_fat`, `is_missing_carbs`, `is_missing_protein`, `is_missing_sodium`

**Total Feature Space**: 1,192 dimensions

## Domain-Specific Rules

The system incorporates rule-based heuristics to complement machine learning predictions:

### Deterministic Hard Rules (Zero False Positive Tolerance)

These rules automatically flag violations of fundamental nutrition data logic:

1. **Added sugars > Total sugars**: `added_sugars > total_sugars + 0.5`
2. **Sugars > Carbs**: `total_sugars > carbs + 1.0`
3. **Fiber > Carbs**: `dietary_fiber > carbs + 1.0`
4. **Saturated fat > Total fat**: `sat_fat > total_fat + 0.5`
5. **Trans fat > Total fat**: `trans_fat > total_fat + 0.5`
6. **Unsaturated fat > Total fat**: `unsat_fat > total_fat + 0.5`
7. **Fat sum > Total fat**: `(sat_fat + trans_fat + unsat_fat) > total_fat + 1.0`

Tolerance values (+0.5 or +1.0) account for rounding errors. These rules catch column swaps, parsing bugs, and unit errors with minimal risk of false positives.

### Sodium Threshold Rule

Products exceeding 1500mg sodium per serving are automatically flagged as anomalies, regardless of model prediction. This deterministic rule ensures regulatory compliance and captures known high-risk patterns.

**Performance on Test Data:**
- Errors dataset: 10 items flagged by sodium rule
- Approved dataset: 9 items flagged by sodium rule
- Hard rules: 29 items flagged across 7 deterministic checks

## Usage

### Pipeline Execution

1. **Data Extraction:**
   ```bash
   cd DataOps && python addcaret.py
   ```

2. **Data Cleaning and Merging:**
   ```bash
   python mergedata.py
   ```

3. **Feature Embedding:**
   ```bash
   python embedtext.py
   ```

4. **Hyperparameter Optimization (Optional):**
   ```bash
   cd ../ModelTraining && python optimizemodel.py
   ```

5. **Model Training:**
   ```bash
   python trainmodel.py
   ```

6. **Model Evaluation:**
   ```bash
   cd ../ModelTests && python testmodel.py
   ```

### Inference API

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# Load ensemble models
models_dir = Path("models")
model_smote = xgb.XGBClassifier()
model_smote.load_model(str(models_dir / "bitecheckSMOTE_V1.json"))

model_adasyn = xgb.XGBClassifier()
model_adasyn.load_model(str(models_dir / "bitecheckADASYN_V1.json"))

# Generate predictions
probs_smote = model_smote.predict_proba(X)[:, 1]
probs_adasyn = model_adasyn.predict_proba(X)[:, 1]
probs_ensemble = np.maximum(probs_smote, probs_adasyn)

# Apply decision threshold
preds = (probs_ensemble >= 0.2).astype(int)

# Apply domain-specific rules
# (See testmodel.py for implementation)
```

## Design Rationale

1. **Supervised Learning from Corrections**: The system learns from human-annotated corrections, using the "Before" state as features and the "After" state to derive labels, capturing the data quality improvement process.

2. **Semantic Text Encoding**: SentenceTransformer embeddings enable the model to understand semantic relationships in product descriptions, improving generalization beyond exact text matching.

3. **Feature Engineering**: Engineered features capture nutrition data consistency patterns, helping identify logical violations and data quality issues that may not be apparent from raw values alone.

4. **Ensemble Methodology**: Combining SMOTE and ADASYN models leverages complementary oversampling strategies, with maximum probability aggregation prioritizing recall for anomaly detection.

5. **Threshold Calibration**: A probability threshold of 0.2 (rather than the default 0.5) balances recall and precision for the operational context, achieving 88.33% recall with 9.15% false positive rate.

6. **Hybrid Approach**: Machine learning predictions are supplemented with rule-based heuristics to ensure comprehensive coverage of known anomaly patterns, including deterministic checks for impossible nutrition values.

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- xgboost >= 2.0.0
- scikit-learn >= 1.3.0
- sentence-transformers >= 2.2.0
- imbalanced-learn >= 0.11.0 (SMOTE/ADASYN implementations)
- optuna >= 3.0.0 (hyperparameter optimization)
- pyarrow >= 12.0.0 (Parquet file I/O)
- openpyxl >= 3.1.0 (Excel file processing)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Performance Characteristics

The ensemble system demonstrates effective performance on imbalanced classification tasks, achieving **88.33% recall** on known error cases while maintaining a **9.15% false positive rate** suitable for production deployment. The maximum probability ensemble strategy provides robustness through model diversity, with complementary oversampling techniques capturing different aspects of the anomaly detection problem space.

Threshold selection represents a fundamental trade-off between recall and precision. Lower thresholds increase anomaly detection coverage but may elevate false positive rates. The implemented threshold (0.2) balances these objectives for the target application domain, successfully meeting the <10% false positive requirement.

Rule-based components complement machine learning predictions by encoding domain expertise and regulatory requirements, ensuring deterministic handling of known high-risk patterns. The hard rules system flags logical violations with near-zero false positive risk, catching data entry errors that may escape machine learning detection.

## Results Summary

- **Training Set**: 17,723 samples, 6.09% anomaly rate
- **Feature Count**: 1,192 features (1,152 embeddings + 13 nutrients + 27 engineered)
- **Test Performance**: 88.33% recall, 9.15% false positive rate
- **Ensemble Strategy**: Maximum probability aggregation
- **Decision Threshold**: 0.2 (optimized for operational requirements)
- **Hard Rules**: 7 deterministic checks for impossible nutrition values

