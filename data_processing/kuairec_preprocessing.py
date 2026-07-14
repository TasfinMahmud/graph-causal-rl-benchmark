import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import ast

def process_social_graph(social_df):
    """
    Converts KuaiRec's friend_list string into a standard edge list format (source, target).
    """
    edges = []
    for _, row in social_df.iterrows():
        user_u = row['user_id']
        # The friend_list is likely a string like "[1, 2, 3]" or a string of ints
        try:
            # Safely evaluate the string representation of the list if it's bracketed
            if isinstance(row['friend_list'], str):
                # some csvs might just have comma separated without brackets, let's be robust
                clean_str = row['friend_list'].strip('[]')
                if not clean_str:
                    continue
                friends = [int(x.strip()) for x in clean_str.split(',') if x.strip()]
            else:
                friends = row['friend_list'] if isinstance(row['friend_list'], list) else []
                
            for user_v in friends:
                edges.append((user_u, user_v))
        except Exception as e:
            continue
            
    edge_df = pd.DataFrame(edges, columns=['source', 'target'])
    return edge_df

def preprocess_kuairec(raw_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths (adjusting for zip extraction structure 'KuaiRec 2.0/data')
    data_dir = os.path.join(raw_dir, "KuaiRec 2.0", "data")
    if not os.path.exists(data_dir):
        # Fallback if unzipped differently
        data_dir = raw_dir
        
    print(f"Loading KuaiRec data from {data_dir}...")
    
    # 1. Process Interaction Matrix (small_matrix)
    print("Processing small_matrix...")
    matrix_df = pd.read_csv(os.path.join(data_dir, "small_matrix.csv"))
    
    # Binarize Reward
    matrix_df['reward'] = (matrix_df['watch_ratio'] >= 2.0).astype(int)
    reward_rate = matrix_df['reward'].mean() * 100
    print(f"Reward Binarization: {reward_rate:.2f}% of interactions are positive (watch_ratio >= 2.0)")
    
    # Temporal Split (70/30)
    print("Performing temporal train/val split on interactions...")
    matrix_df = matrix_df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(matrix_df) * 0.7)
    train_matrix = matrix_df.iloc[:split_idx].copy()
    val_matrix = matrix_df.iloc[split_idx:].copy()
    
    assert train_matrix['timestamp'].max() <= val_matrix['timestamp'].min(), "Temporal Leakage!"
    print(f"Train matrix: {len(train_matrix)} interactions")
    print(f"Val matrix: {len(val_matrix)} interactions")
    
    # 2. Process User Features
    print("\nProcessing user features...")
    user_feat_df = pd.read_csv(os.path.join(data_dir, "user_features.csv"))
    
    continuous_cols = ['follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days']
    # Some cols might be missing in some versions, check first
    cols_to_scale = [c for c in continuous_cols if c in user_feat_df.columns]
    
    if cols_to_scale:
        scaler = StandardScaler()
        user_feat_df[cols_to_scale] = scaler.fit_transform(user_feat_df[cols_to_scale])
        print(f"Z-score normalized features: {cols_to_scale}")
        
    # 3. Process Social Network Graph
    print("\nProcessing social network graph into edge list...")
    social_df = pd.read_csv(os.path.join(data_dir, "social_network.csv"))
    edge_df = process_social_graph(social_df)
    print(f"Extracted {len(edge_df)} directed friendship edges.")
    
    # 4. Export to Parquet
    print("\nExporting processed data to Parquet...")
    train_matrix.to_parquet(os.path.join(output_dir, "train_matrix.parquet"), index=False)
    val_matrix.to_parquet(os.path.join(output_dir, "val_matrix.parquet"), index=False)
    user_feat_df.to_parquet(os.path.join(output_dir, "user_features.parquet"), index=False)
    edge_df.to_parquet(os.path.join(output_dir, "social_edges.parquet"), index=False)
    
    print(f"Success! Stage 2 KuaiRec data saved to {output_dir}")

if __name__ == "__main__":
    RAW_DIR = r"../data/raw/kuairec"
    OUTPUT_DIR = r"../data/processed/kuairec"
    preprocess_kuairec(RAW_DIR, OUTPUT_DIR)
