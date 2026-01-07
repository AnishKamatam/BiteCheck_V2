import pandas as pd
import numpy as np


# Column definitions
ID_COL = ["gtin"]

TEXT_COLS = ["category_name", "product_name", "ingredients_text"]

NUTRIENT_COLS = [
    "calories", "total_fat", "sat_fat", "trans_fat", "unsat_fat",
    "cholesterol", "sodium", "carbs", "dietary_fiber",
    "total_sugars", "added_sugars", "protein", "potassium",
]

FEATURE_COLS = [
    # Calorie consistency
    "expected_calories", "calorie_gap", "calorie_gap_ratio",
    # Macro calorie shares
    "fat_calorie_share", "carb_calorie_share", "protein_calorie_share", "macro_share_sum",
    # Sugar logic
    "added_sugar_ratio", "sugar_carb_ratio", "added_sugars_gt_total", "sugars_gt_carbs",
    # Fat structure
    "fat_sum", "fat_gap", "fat_gap_ratio",
    # Density / intensity
    "sodium_per_calorie", "sugar_per_calorie", "protein_per_calorie",
    # Zero / missing flags
    "is_zero_calories", "is_zero_total_fat", "is_zero_carbs", "is_zero_protein", "is_zero_sodium",
    "is_missing_calories", "is_missing_total_fat", "is_missing_carbs", "is_missing_protein", "is_missing_sodium",
]

COLS_TO_KEEP = ID_COL + TEXT_COLS + NUTRIENT_COLS + FEATURE_COLS + ["label_is_anomaly"]


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    # Safe division that handles division by zero and negative denominators
    # Returns 0 when denominator is zero or negative
    return np.where(
        (denominator > 0) & (denominator.notna()) & (numerator.notna()),
        numerator / denominator,
        0
    )


def add_nutrition_features(df: pd.DataFrame) -> pd.DataFrame:
    # Adds engineered nutrition features to help identify inconsistencies and anomalies
    df = df.copy()

    # Calculate expected calories from macros (fat: 9 cal/g, carbs/protein: 4 cal/g)
    df["expected_calories"] = (
        9 * df["total_fat"] +
        4 * df["carbs"] +
        4 * df["protein"]
    )
    # Find discrepancy between reported and calculated calories
    df["calorie_gap"] = (df["calories"] - df["expected_calories"]).abs()
    df["calorie_gap_ratio"] = safe_div(df["calorie_gap"], df["calories"])

    # Calculate what percentage of calories come from each macro
    df["fat_calorie_share"] = safe_div(9 * df["total_fat"], df["calories"])
    df["carb_calorie_share"] = safe_div(4 * df["carbs"], df["calories"])
    df["protein_calorie_share"] = safe_div(4 * df["protein"], df["calories"])
    # Sum should be close to 1.0 if data is consistent
    df["macro_share_sum"] = (
        df["fat_calorie_share"] +
        df["carb_calorie_share"] +
        df["protein_calorie_share"]
    )

    # Added sugars should be less than or equal to total sugars
    df["added_sugar_ratio"] = safe_div(
        df["added_sugars"], df["total_sugars"]
    )
    # Sugars should be less than or equal to carbs
    df["sugar_carb_ratio"] = safe_div(
        df["total_sugars"], df["carbs"]
    )
    # Flag violations where added sugars exceed total sugars
    df["added_sugars_gt_total"] = (df["added_sugars"] > df["total_sugars"]).astype(int)
    # Flag violations where sugars exceed carbs
    df["sugars_gt_carbs"] = (df["total_sugars"] > df["carbs"]).astype(int)

    # Sum of fat subtypes should equal total fat
    df["fat_sum"] = (
        df["sat_fat"] +
        df["trans_fat"] +
        df["unsat_fat"]
    )
    df["fat_gap"] = (df["total_fat"] - df["fat_sum"]).abs()
    df["fat_gap_ratio"] = safe_div(df["fat_gap"], df["total_fat"])

    # Nutrient density per calorie (useful for identifying outliers)
    df["sodium_per_calorie"] = safe_div(df["sodium"], df["calories"])
    df["sugar_per_calorie"] = safe_div(df["total_sugars"], df["calories"])
    df["protein_per_calorie"] = safe_div(df["protein"], df["calories"])

    # Flag zero and missing values (we use -1 for missing in cleaning)
    for col in ["calories", "total_fat", "carbs", "protein", "sodium"]:
        df[f"is_zero_{col}"] = (df[col] == 0).astype(int)
        df[f"is_missing_{col}"] = (df[col] < 0).astype(int)

    return df


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    # Cleans and preprocesses dataframe for model training
    initial_count = len(df)
    

    # Keep only target columns
    keep = [c for c in COLS_TO_KEEP if c in df.columns]
    df = df[keep].copy()


    # Convert ID and text columns to string
    for col in ID_COL + TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")


    # Create label_is_anomaly from carets in original data (before cleaning)
    df["label_is_anomaly"] = 0
    cols_with_carets = TEXT_COLS + NUTRIENT_COLS 
    for col in cols_with_carets:
        if col in df.columns:
            col_str = df[col].astype(str).str.strip()
            has_caret = col_str.fillna("").str.endswith("^")
            df["label_is_anomaly"] = df["label_is_anomaly"] | has_caret.astype(int)


    # Convert nutrient columns to numeric (remove carets, fill NaN with -1)
    for col in NUTRIENT_COLS:
        if col in df.columns:
            numeric_val = df[col].astype(str).str.rstrip("^").str.strip()
            df[col] = pd.to_numeric(numeric_val, errors="coerce")
            df[col] = df[col].fillna(-1)



    # Remove rows without GTIN
    before_gtin = len(df)
    df = df.dropna(subset=["gtin"])
    removed_no_gtin = before_gtin - len(df)


    # Remove rows with empty ingredients_text
    removed_empty_ingredients = 0
    if "ingredients_text" in df.columns:
        before_ingredients = len(df)
        df = df[
            df["ingredients_text"].notna() & 
            (df["ingredients_text"].astype(str).str.strip() != "")
        ]
        removed_empty_ingredients = before_ingredients - len(df)


    # Remove duplicate GTINs (keep first occurrence)
    before_duplicates = len(df)
    df = df.sort_values("gtin")
    df = df.drop_duplicates(subset=["gtin"], keep="first")
    removed_duplicates = before_duplicates - len(df)


    final_count = len(df)
    total_removed = initial_count - final_count


    if verbose:
        print(f"  Initial rows: {initial_count:,}")
        print(f"  Removed (no GTIN): {removed_no_gtin:,}")
        print(f"  Removed (empty ingredients_text): {removed_empty_ingredients:,}")
        print(f"  Removed (duplicate GTINs): {removed_duplicates:,}")
        print(f"  Final rows: {final_count:,}")
        print(f"  Total removed: {total_removed:,}")

    # Add engineered nutrition features
    df = add_nutrition_features(df)

    return df
