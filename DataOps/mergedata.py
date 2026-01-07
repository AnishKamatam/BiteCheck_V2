import pandas as pd
from pathlib import Path
from cleandata import clean


def load_and_merge_data():
    # Loads, cleans, and merges Before/After datasets
    data_folder = Path(__file__).parent.parent / "Data" / "PreOpDataCSV"
    
    # Load and clean all Before files
    before_files = sorted(data_folder.glob("Before*.csv"))
    before_cleaned = []
    for file in before_files:
        df = pd.read_csv(file, low_memory=False)
        before_cleaned.append(clean(df, verbose=False))
    

    # Load and clean all After files
    after_files = sorted(data_folder.glob("After*.csv"))
    after_cleaned = []
    for file in after_files:
        df = pd.read_csv(file, low_memory=False)
        after_cleaned.append(clean(df, verbose=False))
    

    # Merge all cleaned datasets
    before_all_merged = pd.concat(before_cleaned, ignore_index=True)
    after_all_merged = pd.concat(after_cleaned, ignore_index=True)
    

    # Merge: features from Before, label from After
    before_features = before_all_merged.drop(columns=["label_is_anomaly"], errors="ignore")
    after_label = after_all_merged[["gtin", "label_is_anomaly"]]
    merged = pd.merge(before_features, after_label, on="gtin", how="inner")
    

    # Save merged datasets
    before_all_merged.to_csv(data_folder / "before_all_cleaned.csv", index=False)
    after_all_merged.to_csv(data_folder / "after_all_cleaned.csv", index=False)
    merged.to_csv(data_folder / "merged.csv", index=False)
    
    
    print(f"Saved merged datasets:")
    print(f"  - before_all_cleaned.csv: {len(before_all_merged):,} rows")
    print(f"  - after_all_cleaned.csv: {len(after_all_merged):,} rows")
    print(f"  - merged.csv: {len(merged):,} rows")
    
    return before_all_merged, after_all_merged, merged


if __name__ == "__main__":
    load_and_merge_data()
