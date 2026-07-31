import os
import ctypes

os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "sequential" 

try:
    ctypes.CDLL("libmkl_rt.so", mode=ctypes.RTLD_GLOBAL)
except:
    pass

import math
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm

from torch.utils.data import DataLoader

import faiss
from src.utils import get_logger, get_featurizer, set_random_seed
from src.data import get_task_dir, DTADataModule
import src.model as module_arch
from src.featurizers.molecule import MolGraphFeaturizer
from src.featurizers.protein import ProteinGNNFeaturizer

logg = get_logger()
captured_features = {}

def get_activation(name):
    def hook(model, input, output):
        captured_features[name] = output.detach().cpu()
    return hook

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')

def str_to_list(arg):
    return arg.split(',')

@torch.no_grad()
def extract_features_and_preds(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_feats = [], [], []
    for batch in tqdm(dataloader, desc="Feature Extraction"):
        batch = batch.to(device)
        preds = model(batch)
        all_preds.append(preds.cpu())
        all_labels.append(batch.y.cpu())
        if 'v_cat' in captured_features:
            all_feats.append(captured_features['v_cat'].clone())
            captured_features.clear()
    return torch.cat(all_preds), torch.cat(all_labels).squeeze(), torch.cat(all_feats)

def calculate_interval_score(y_true, y_preds, q_low, q_high, alpha):

    lower = y_preds + q_low
    upper = y_preds + q_high
    width = upper - lower
    

    under = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    over = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    
    score = width + under + over
    return score.mean()

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--task", choices=["KIBA", "Davis"], required=True)
    parser.add_argument("--use-cold-spilt", type=str2bool, default=False)
    parser.add_argument("--cold", type=str_to_list, default=['Drug','target_key'])
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=42)
    args = parser.parse_args()

    base_pth = "/home/zhaol409100220027/zhaol409100220027/lee409100240016/CPDTA1/bestmodel/"
    model_map = {
        "Davis": base_pth + "CPDTA_Davis.pth",
        "KIBA_random": base_pth + "CPDTA_KIBA.pth",
        "KIBA_cold_drug": base_pth + "CPDTA_colddrug.pth",
        "KIBA_cold_target": base_pth + "CPDTA_coldtarget.pth",
        "KIBA_cold_drug_target": base_pth + "CPDTA_coldall.pth"
    }
    
    if args.task == "Davis":
        task_key = "Davis"
    else:
        if not args.use_cold_spilt: task_key = "KIBA_random"
        else:
            clean_cold = sorted([c.lower().replace('target_key', 'target') for c in args.cold])
            task_key = f"KIBA_cold_{'_'.join(clean_cold)}"

    checkpoint_path = model_map.get(task_key)
    config = OmegaConf.load(args.config)
    device = torch.device(f"cuda:{args.device}")
    out_dir = "result_cp_resampled"
    os.makedirs(out_dir, exist_ok=True)

    model = module_arch.DTAPredictor(
        drug_node_dim=78, drug_pre_dim=512, prot_node_dim=61, prot_pre_dim=1280,
        drug_dim=256, target_dim=256, use_fusion=True
    ).to(device).float()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.fused_norm.register_forward_hook(get_activation('v_cat'))

    task_dir = get_task_dir(args.task)
    set_random_seed(args.train_seed)
    datamodule = DTADataModule(
        data_dir=task_dir, drug_seq_featurizer=get_featurizer(config.drug_seq_featurizer, save_dir=task_dir),
        target_seq_featurizer=get_featurizer(config.target_seq_featurizer, save_dir=task_dir),
        drug_struc_featurizer=MolGraphFeaturizer(save_dir=task_dir), 
        target_struc_featurizer=ProteinGNNFeaturizer(save_dir=task_dir),
        dataset_name=args.task, batch_size=args.batch_size, shuffle=False,
        use_cold_spilt=args.use_cold_spilt, cold=args.cold
    )
    datamodule.prepare_data()
    datamodule.setup()
    
    cal_loader, _ = datamodule.cal_dataloader()
    test_loader, _ = datamodule.test_dataloader()

    all_summary_results = []
    
    logg.info(f"\n>>> Evaluating on original split with Train Seed: {args.train_seed}")

    cal_preds, cal_labels, cal_feats = extract_features_and_preds(model, cal_loader, device)
    test_preds, test_labels, test_feats = extract_features_and_preds(model, test_loader, device)

    test_preds_np = test_preds.numpy()
    test_labels_np = test_labels.numpy()
    cal_res_abs = torch.abs(cal_labels - cal_preds).numpy()
    cal_errors_sorted = np.sort((cal_labels - cal_preds).numpy())
    
    alphas = [0.05, 0.1, 0.2]
    K_ratios = [0.05, 0.1, 0.3, 0.5, 0.75, 1.0]

    for alpha in alphas:
        low_idx = max(0, math.floor((len(cal_labels) + 1) * (alpha / 2)) - 1)
        high_idx = min(len(cal_labels) - 1, math.ceil((len(cal_labels) + 1) * (1 - alpha / 2)) - 1)
        q_low, q_high = cal_errors_sorted[low_idx], cal_errors_sorted[high_idx]
        
        cov = ((test_labels_np >= (test_preds_np + q_low)) & (test_labels_np <= (test_preds_np + q_high))).mean()
        score = calculate_interval_score(test_labels_np, test_preds_np, q_low, q_high, alpha)
        
        all_summary_results.append({
            "Alpha": alpha, "Method": "CQR", 
            "Empirical_Coverage": cov, "Avg_Width": q_high - q_low, "Interval_Score": score
        })

    cal_feats_norm = torch.nn.functional.normalize(cal_feats, p=2, dim=1).numpy().astype('float32')
    test_feats_norm = torch.nn.functional.normalize(test_feats, p=2, dim=1).numpy().astype('float32')
    index = faiss.IndexFlatIP(cal_feats_norm.shape[1])
    index.add(cal_feats_norm)

    for ratio in K_ratios:
        K = max(1, int(len(cal_labels) * ratio))
        _, I = index.search(test_feats_norm, K)
        batch_knn_res = np.sort(cal_res_abs[I], axis=1)
        
        method_name = "CP" if ratio == 1.0 else f"RCP_{ratio}"
        
        for alpha in alphas:
            q_idx = min(math.ceil((K + 1) * (1 - alpha)) - 1, K - 1)
            q_values = batch_knn_res[:, q_idx]
            
            emp_cov = ((test_labels_np >= (test_preds_np - q_values)) & 
                       (test_labels_np <= (test_preds_np + q_values))).mean()

            score = calculate_interval_score(test_labels_np, test_preds_np, -q_values, q_values, alpha)
            
            all_summary_results.append({
                "Alpha": alpha, "Method": method_name, 
                "Empirical_Coverage": emp_cov, "Avg_Width": q_values.mean() * 2, "Interval_Score": score
            })


    final_df = pd.DataFrame(all_summary_results)

    metrics = ['Empirical_Coverage', 'Avg_Width', 'Interval_Score']
    for m in metrics:
        final_df[m] = final_df[m].apply(lambda x: f"{x:.4f}")


    method_order = ['RCP_0.05', 'RCP_0.1', 'RCP_0.3', 'RCP_0.5', 'RCP_0.75', 'CP', 'CQR']
    final_df['Method'] = pd.Categorical(final_df['Method'], categories=method_order, ordered=True)
    final_df = final_df.sort_values(['Alpha', 'Method'])
    

    final_df['Alpha'] = final_df['Alpha'].astype(str)
    final_df.loc[final_df['Alpha'] == final_df['Alpha'].shift(1), 'Alpha'] = ''

    final_csv = f"{out_dir}/Final_Stats_{task_key}.csv"
    final_df.to_csv(final_csv, index=False)
    logg.info(f"Complete. Results saved in {final_csv}")

if __name__ == "__main__":
    main()
