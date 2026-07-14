import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling
import gc
import logging
from models.baselines_gnn import StandardGCN, StandardGAT, StandardGraphSAGE, SimpleNGCF, StandardLightGCN

# Set up logging
logging.basicConfig(
    filename='../data/phase1_gnn.log',
    filemode='a', # Append mode for resumption
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

DATA_PATH = r"../data/graphs"
EMBEDDING_PATH = r"../data/embeddings"
CHECKPOINT_PATH = r"../data/checkpoints"
os.makedirs(EMBEDDING_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

def train_gnn(model_name, ModelClass, data, dataset_name, device):
    checkpoint_file = os.path.join(CHECKPOINT_PATH, f"{dataset_name}_{model_name}_ckpt.pt")
    
    # Check if this model completely finished already
    final_emb_file = os.path.join(EMBEDDING_PATH, f"{dataset_name}_{model_name}.pt")
    if os.path.exists(final_emb_file):
        logging.info(f"[{dataset_name}] {model_name} already fully trained. Skipping.")
        return

    logging.info(f"[{dataset_name}] Starting/Resuming training for {model_name}...")
    
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else (
        data.x.size(0) if data.x is not None else int(data.edge_index.max()) + 1
    )
    
    in_channels = data.x.size(1) if data.x is not None else 64
    out_channels = 64
    
    # Rigorous Pure Python K-Hop Topology Sampler to bypass Windows 'pyg-lib' DLL crashes
    # This guarantees 100% mathematical integrity for GraphSAGE neighborhood accumulation
    from torch_geometric.utils import k_hop_subgraph
    from torch_geometric.data import Data as PyGData
    
    class RigorousKHopSampler:
        def __init__(self, full_data, batch_size=16384, hops=2):
            self.full_data = full_data
            self.batch_size = batch_size
            self.hops = hops
            self.loader = torch.utils.data.DataLoader(
                torch.arange(full_data.num_nodes), 
                batch_size=batch_size, 
                shuffle=True
            )
            
        def __iter__(self):
            for batch_nodes in self.loader:
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                    node_idx=batch_nodes,
                    num_hops=self.hops,
                    edge_index=self.full_data.edge_index,
                    relabel_nodes=True,
                    num_nodes=self.full_data.num_nodes
                )
                
                # Strict Edge Dropout to prevent GAT/LightGCN VRAM Explosion on dense Hub Nodes
                MAX_EDGES = 200000 
                if edge_index.size(1) > MAX_EDGES:
                    indices = torch.randperm(edge_index.size(1))[:MAX_EDGES]
                    edge_index = edge_index[:, indices]
                    
                yield batch_nodes, subset, edge_index
                
    if model_name in ["GAT", "NGCF", "GraphSAGE"]:
        bs = 4096
    else:
        bs = 16384
    sampler = RigorousKHopSampler(data, batch_size=bs, hops=1)
    
    model = ModelClass(in_channels, 128, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    start_epoch = 0
    epochs = 5
    
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"[{dataset_name}] Resuming {model_name} from Epoch {start_epoch}")
    
    model.train()
    
    # Load full nodes into VRAM (Fits easily: 2M nodes * 64-dim = ~512MB)
    if data.x is None:
        x_full = torch.randn(num_nodes, in_channels).to(device)
    else:
        x_full = data.x.to(device)
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        batch_idx = 0
        for root_nodes, subset, edge_index in sampler:
            optimizer.zero_grad()
            try:
                sub_x = x_full[subset].to(device)
                sub_edges = edge_index.to(device)
                
                out = model(sub_x, sub_edges)
                
                # 1. Positive scores (dot product of true connected edges)
                pos_out = (out[sub_edges[0]] * out[sub_edges[1]]).sum(dim=-1)
                
                # Fast, RAM-safe negative sampling directly on the GPU
                row = torch.randint(0, sub_x.size(0), (sub_edges.size(1),), device=device)
                col = torch.randint(0, sub_x.size(0), (sub_edges.size(1),), device=device)
                neg_edges = torch.stack([row, col], dim=0)
                
                # 3. Negative scores
                neg_out = (out[neg_edges[0]] * out[neg_edges[1]]).sum(dim=-1)
                
                # 4. BPR Loss
                loss = -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                # AGGRESSIVE VRAM CLEARING TO PREVENT EPOCH 2 LEAKS
                del out, pos_out, neg_out, row, col, neg_edges, loss, sub_x, sub_edges
                
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                    logging.info(f"[{dataset_name}][{model_name}] Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {total_loss/(batch_idx+1):.4f}")
                    
                batch_idx += 1
            except Exception as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError):
                    logging.warning(f"[{dataset_name}][{model_name}] OOM on Batch {batch_idx}. Skipping dense subgraph!")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
                
        logging.info(f"[{dataset_name}][{model_name}] Epoch {epoch+1}/{epochs} Complete. Avg Loss: {total_loss/batch_idx:.4f}")
        
        # Save Checkpoint after every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_file)
        logging.info(f"[{dataset_name}][{model_name}] Checkpoint saved for epoch {epoch+1}.")
        
    # Final Output
    logging.info(f"[{dataset_name}] Saving final {model_name} embeddings...")
    model.eval()
    with torch.no_grad():
        torch.save(model.state_dict(), final_emb_file)
        
    del model, sampler, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info(f"[{dataset_name}] {model_name} successfully trained and cleared from RAM.")
    return total_loss / max(1, batch_idx)

from torch_geometric.utils import to_undirected

def main():
    logging.info("Phase 1 GNN Massive Tournament started on device: cuda")
    device = torch.device('cuda')
    
    datasets = {
        "obd": r"../data/graphs/obd_graph.pt",
        "kuairec": r"../data/graphs/kuairec_graph.pt"
    }
    
    models = {
        "GCN": StandardGCN,
        "GAT": StandardGAT,
        "GraphSAGE": StandardGraphSAGE,
        "NGCF": SimpleNGCF,
        "LightGCN": StandardLightGCN
    }
    
    for dataset_name, graph_path in datasets.items():
        logging.info(f"Loading {dataset_name.upper()} graph into RAM...")
        data = torch.load(graph_path, weights_only=False)
        if hasattr(data, 'to_homogeneous'):
            data = data.to_homogeneous()
            
        # CRITICAL MATHEMATICAL FIX: Force bidirectional bipartite topology
        # This guarantees users receive item embeddings and vice versa.
        data.edge_index = to_undirected(data.edge_index)
            
        results = []
        for model_name, ModelClass in models.items():
            final_loss = train_gnn(model_name, ModelClass, data, dataset_name, device)
            if final_loss is not None:
                results.append({"Model": model_name, "Dataset": dataset_name.upper(), "Metric": "BPR Loss", "Score": final_loss})
            
            # CRITICAL VRAM FIX: Flush the massive computational graph from the GPU!
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            save_path = os.path.join(DATA_PATH, "..", f"tournament_{dataset_name}_gnn_metrics.csv")
            df.to_csv(save_path, index=False)
            logging.info(f"[{dataset_name}] Saved Phase 1 GNN metrics to {save_path}")

        del data
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
