import pandas as pd
import numpy as np
import hashlib
from sklearn.preprocessing import StandardScaler
import os

def hash_user_features(row):
    """
    Creates a synthetic user ID by hashing the four categorical demographic/context features.
    This groups similar interaction contexts into a single 'User Persona' node.
    """
    feature_str = f"{row['user_feature_0']}_{row['user_feature_1']}_{row['user_feature_2']}_{row['user_feature_3']}"
    return hashlib.md5(feature_str.encode('utf-8')).hexdigest()[:16]

def preprocess_obd(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading OBD data from {data_dir}...")
    men_df = pd.read_csv(os.path.join(data_dir, "men.csv"))
    
    print(f"Original dataset size: {len(men_df)} interactions")
    
    # 2. Synthetic User IDs
    print("Generating synthetic User IDs from demographic features...")
    men_df['user_id'] = men_df.apply(hash_user_features, axis=1)
    
    num_unique_users = men_df['user_id'].nunique()
    print(f"Created {num_unique_users} unique user personas.")
    
    # 3. Propensity Clipping
    print("Clipping propensity scores to minimum 0.01 for IPW stability...")
    men_df['propensity_score'] = men_df['propensity_score'].clip(lower=0.01)
    
    # 4. Position One-Hot Encoding
    print("One-hot encoding display position...")
    # OBD positions are 1, 2, 3
    pos_dummies = pd.get_dummies(men_df['position'], prefix='pos')
    men_df = pd.concat([men_df, pos_dummies], axis=1)
    
    # 5. Normalization of continuous affinity features
    print("Z-score normalizing continuous affinity features...")
    affinity_cols = [c for c in men_df.columns if c.startswith('user-item_affinity_')]
    scaler = StandardScaler()
    men_df[affinity_cols] = scaler.fit_transform(men_df[affinity_cols])
    
    # 6. Temporal Train/Val Split (70/30)
    print("Performing temporal train/validation split (70/30)...")
    # Sort by timestamp
    men_df = men_df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(men_df) * 0.7)
    
    train_df = men_df.iloc[:split_idx].copy()
    val_df = men_df.iloc[split_idx:].copy()
    
    print(f"Train set: {len(train_df)} interactions")
    print(f"Val set: {len(val_df)} interactions")
    
    # Verify no data leakage
    assert train_df['timestamp'].max() <= val_df['timestamp'].min(), "CRITICAL: Temporal data leakage detected!"
    
    # Print Reward Statistics
    print(f"Train mean click rate: {train_df['click'].mean():.4f}")
    print(f"Val mean click rate: {val_df['click'].mean():.4f}")
    
    # 7. Export to Parquet
    print("Exporting processed DataFrames to Parquet...")
    train_path = os.path.join(output_dir, "men_train.parquet")
    val_path = os.path.join(output_dir, "men_val.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    print(f"Success! Data saved to {output_dir}")

if __name__ == "__main__":
    DATA_DIR = r"../data/raw/obd/random/men"
    OUTPUT_DIR = r"../data/processed/obd/random/men"
    preprocess_obd(DATA_DIR, OUTPUT_DIR)
