import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import logging
import gc
import json
from src.baselines_causal import get_grf_model, get_forestdr_model, get_xlearner_model, get_slearner_model
from src.baselines_dragonnet import DragonNet

logging.basicConfig(
    filename='../data/phase2_causal.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

DATA_PATH = r"../data/processed"
RESULTS_PATH = r"../data/causal"
CHECKPOINT_PATH = r"../data/checkpoints"
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

def train_causal(model_name, model, dataset_name):
    checkpoint_file = os.path.join(CHECKPOINT_PATH, f"{dataset_name}_{model_name}_causal_ckpt.json")
    save_path = os.path.join(RESULTS_PATH, f"{dataset_name}_{model_name}_cate.csv")
    # Ensure we bypass missing static CSVs since we are dynamically streaming 26M rows in chunks
    logging.info(f"[{dataset_name}] Utilizing Massive Chunk Streamer for Data Access...")

    # Checkpoint logic
    start_chunk = 0
    is_fitted = False
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            ckpt = json.load(f)
            start_chunk = ckpt.get('last_chunk', 0)
            is_fitted = ckpt.get('is_fitted', False)
            logging.info(f"[{dataset_name}] Resuming {model_name} from Chunk {start_chunk}")
            
    if start_chunk == "DONE":
        logging.info(f"[{dataset_name}] {model_name} is already entirely processed.")
        try:
            df_out = pd.read_csv(save_path)
            variance = df_out['cate'].var()
            del df_out
            gc.collect()
            return variance
        except:
            return 0.0

    logging.info(f"[{dataset_name}] Starting causal chunking for {model_name}...")
    
    chunk_size = 500000
    total_chunks = 52 # 52 * 500k = ~26M rows
    
    def data_streamer():
        csv_path = r"../data/raw/obd/all.csv" if dataset_name == "obd" else r"../data/processed/big_matrix.csv"
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            yield chunk
            
    chunk_idx = 0
    
    for df_chunk in data_streamer():
        if chunk_idx < start_chunk:
            chunk_idx += 1
            continue
            
        logging.info(f"[{dataset_name}][{model_name}] Processing Physical Chunk {chunk_idx} (Rows: {len(df_chunk)})...")
        
        # Rigorous mapping to true Dataset columns
        if dataset_name == "obd":
            features = [c for c in df_chunk.columns if 'user_feature' in c or 'user-item_affinity' in c]
            for col in features:
                try:
                    df_chunk[col] = pd.to_numeric(df_chunk[col])
                except ValueError:
                    df_chunk[col] = df_chunk[col].apply(lambda x: hash(str(x)) % 100000)
            X = df_chunk[features].fillna(0).values if features else np.zeros((len(df_chunk), 1))
            T = df_chunk['position'].values
            Y = df_chunk['click'].values
        else:
            # KuaiRec mapping
            features = ['video_duration', 'timestamp']
            X = df_chunk[features].fillna(0).values
            T = (df_chunk['watch_ratio'] > 1.0).astype(int).values
            Y = df_chunk['play_duration'].values
        
        if not is_fitted:
            logging.info(f"[{dataset_name}] Fitting {model_name} on primary convergence block (Chunk 0)...")
            model.fit(Y, T, X=X)
            is_fitted = True
            
        logging.info(f"[{dataset_name}][{model_name}] Predicting CATE for Chunk {chunk_idx}...")
        
        # Explicitly pass T0 and T1 to prevent EconML from assuming binary [0, 1] treatments
        t_min, t_max = np.min(T), np.max(T)
        cate_preds = model.effect(X, T0=t_min, T1=t_max)
        
        # Append to CSV
        df_out = pd.DataFrame({'cate': cate_preds})
        mode = 'w' if chunk_idx == 0 else 'a'
        header = True if chunk_idx == 0 else False
        df_out.to_csv(save_path, mode=mode, header=header, index=False)
        
        # Save Checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({'last_chunk': chunk_idx + 1, 'is_fitted': is_fitted}, f)
            
        chunk_idx += 1
        
        # Clean RAM strictly per chunk
        del df_chunk, X, T, Y, cate_preds, df_out
        gc.collect()

    with open(checkpoint_file, 'w') as f:
        json.dump({'last_chunk': "DONE", 'is_fitted': True}, f)
        
    logging.info(f"[{dataset_name}] {model_name} CATE scoring completely finished across all chunks.")
    del model
    gc.collect()
    
    # Calculate Variance from the saved CSV for the final table
    try:
        df_out = pd.read_csv(save_path)
        variance = df_out['cate'].var()
        del df_out
        gc.collect()
        return variance
    except:
        return 0.0

def main():
    logging.info("Phase 2 Causal Massive Tournament started.")
    datasets = ["obd", "kuairec"]
    
    for dataset_name in datasets:
        models = {
            "DragonNet": DragonNet(epochs=3, batch_size=2048),
            "SLearner": get_slearner_model(),
            "XLearner": get_xlearner_model(),
            "GRF": get_grf_model()
        }
        
        results = []
        for name, model in models.items():
            variance = train_causal(name, model, dataset_name)
            if variance is not None:
                results.append({"Model": name, "Dataset": dataset_name.upper(), "Metric": "CATE Variance", "Score": variance})
                
        if results:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(RESULTS_PATH, "..", f"tournament_{dataset_name}_causal_metrics.csv"), index=False)
            
if __name__ == "__main__":
    main()
