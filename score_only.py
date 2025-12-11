import os
import pickle
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
import config
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from semantic_baselines import compute_semantic_entropy, compute_deg_semantic_density, pro_score, eigen_score, eigen_score_refactor, eigen_score_origin_v2, compute_eigen_embed, compute_semantic_similarity
from semantic_baselines import MODEL_PATH_DICT
from collections import defaultdict


def compute_weighted_mean(embeddings, weights):
    """Compute weighted mean of embeddings."""
    weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(embeddings.device)
    return (weights * embeddings).sum(dim=0)


def compute_metrics(embeddings, mean, weights, p=1):
    """
    Compute various distance-based metrics between embeddings and a mean embedding.
    
    Args:
        embeddings (torch.Tensor): shape (N, D)
        mean (torch.Tensor): shape (D,)
        weights (array-like): length N, non-negative
        p (float): order for Lp or Wasserstein metric
        method (str): one of {"lp", "wasserstein", "cosine", "mahalanobis"}
        cov_inv (torch.Tensor, optional): inverse covariance for Mahalanobis distance
    """
    device = embeddings.device
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    weights_tensor = weights_tensor / weights_tensor.sum()  # normalize weights
    diffs = embeddings - mean

    norms = torch.norm(diffs, p=p, dim=1)
    weighted = (weights_tensor * norms).sum()
    return weighted.item()

# ----------------
### Load LLM model
# ----------------
def load_model_from_path(model_name, device):
    if model_name not in MODEL_PATH_DICT:
        raise ValueError(f"Model {model_name} not supported")
    model_path = MODEL_PATH_DICT[model_name].lower()

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
        use_fast=False if "falcon3" in model_path else True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- dtype & model class ---
    if "gemma3" in model_path:
        from transformers import Gemma3ForCausalLM
        model_cls = Gemma3ForCausalLM
    else:
        model_cls = AutoModelForCausalLM

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        cache_dir=config.hf_cache_dir
    )

    return model, tokenizer

import torch.nn.functional as F
#### -------------------- 20251205 --------------------
def main(args, semantic_model, semantic_tokenizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer(args.embed_model).to(device)

    # --- Load generations ---
    if args.n_samples == 10:
        gen_path = f"{config.output_dir}/{args.dataset}__{args.model}__generation.pkl"
    else:
        gen_path = f"{config.output_dir}/{args.dataset}__{args.model}__{args.n_samples}__generation.pkl"
    with open(gen_path, "rb") as infile:
        generations = pickle.load(infile)

    labels = []
    norm_dict = defaultdict(list)

    for i, gen in enumerate(tqdm(generations, desc="Processing generations")):
        # --- Label ---
        if args.dataset in ['gsm8k', 'svamp', 'arith']:
            label = 1 - int(gen['eval_score'] == 1.0)
        else:
            label = 1 - int(gen['eval_score'] > 0.3)
        labels.append(label)

        cleaned_texts = gen["cleaned_generated_texts"]
        samples_avg_nll = gen["samples_avg_nll"]

        # --- Weighting sets ---
        probs = np.exp(-np.array(samples_avg_nll))
        probs /= probs.sum()

        # --- Embeddings ---
        embeddings = embed_model.encode(cleaned_texts, convert_to_tensor=True, device=device)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # --- Base version ---
        mean_embedding = torch.mean(embeddings, dim=0)
        diffs = embeddings - mean_embedding 
        ot_scores = torch.norm(diffs, p=1, dim=1)
        
        # --- Weighted version ---
        probs = np.exp(-np.array(samples_avg_nll))
        probs /= probs.sum() 
        weighted_mean_embeddings = compute_weighted_mean(
                embeddings, torch.tensor(probs, dtype=torch.float32, device=device)
        )        
        diffs_weighted = embeddings - weighted_mean_embeddings.unsqueeze(0)
        weighted_ot_scores = torch.norm(diffs_weighted, p=1, dim=1)

        eigen_embed = compute_eigen_embed(embeddings, alpha=1e-3)
        
        # --- Store norms ---
        norm_dict["EigenEmbed"].append(eigen_embed)
        norm_dict["OT Score (base)"].append(ot_scores.sum().item())
        norm_dict["OT Score (weighted)"].append((torch.tensor(probs, dtype=torch.float32, device=device) 
                                                 * weighted_ot_scores).sum().item())
        
        ### PRO
        norm_dict["PRO"].append(pro_score(gen))
        
        ### Semantic Baselines
        if args.semantic_baselines:
            sem_entropy = compute_semantic_entropy(gen, semantic_model, semantic_tokenizer)
            norm_dict["semantic_entropy"].append(sem_entropy)

            deg_val, sd_val = compute_deg_semantic_density(gen, semantic_model, semantic_tokenizer)
            norm_dict["deg"].append(deg_val)
            norm_dict["semantic_density"].append(sd_val)

    # --- AUROC reporting ---
    results = {}
    print("\n=== Metric Performance (ROC-AUC) ===")
    for method, values in norm_dict.items():
        auc = roc_auc_score(labels, values)
        results[method] = round(auc, 4)
        print(f"{method:45s} → ROC-AUC: {auc:.4f}")
    print("====================================\n")
    
    json_path = f"{config.result_dir}/{args.dataset}__{args.model}__{args.n_samples}__{args.embed_model}__{args.version}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    return results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute uncertainty scores for generated sequences')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--embed_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--prob_score', type=str, default='avg_nll', help='Probability score to use for weighting (nll or avg_nll)')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--eigen_baselines', type=bool, default=False, help='Whether to compute eigen baselines')
    parser.add_argument('--eigen_embed_only', type=bool, default=False, help='Whether to compute only eigen embedding baselines')
    parser.add_argument('--version', type=str, default='20251205', help='Version identifier for the run')
    parser.add_argument('--semantic_baselines', type=bool, default=True, help='Whether to compute semantic baselines')
    args = parser.parse_args()
    
    # --- Load semantic model and tokenizer ---
    semantic_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    semantic_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to('cuda')
    
    print(f"Running scoring for {args.dataset} with model {args.model} using {args.embed_model} embeddings, n_samples:{args.n_samples}.")
    main(args, semantic_model, semantic_tokenizer)
