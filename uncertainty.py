import warnings
warnings.filterwarnings('ignore')

import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
import config
import json
import argparse
from collections import defaultdict
from collections import Counter
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import (
    set_seed,
    compute_weighted_mean,
    compute_semantic_entropy, 
    compute_deg_semantic_density, 
    pro_score, 
    compute_eigen_embed
    )

#### -------------------- SCORING --------------------
def main(args, semantic_model, semantic_tokenizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer(args.embed_model).to(device)

    # --- Load generations ---
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
        
        # --- RDS base version ---
        mean_embedding = torch.mean(embeddings, dim=0)
        diffs = embeddings - mean_embedding 
        rds_scores = torch.norm(diffs, p=1, dim=1)
        rds = rds_scores.sum().item()
        rds_l2 = torch.norm(diffs, p=2, dim=1).sum().item()
        
        # --- RDS weighted version ---
        probs = np.exp(-np.array(samples_avg_nll))
        probs /= probs.sum() 
        weighted_mean_embeddings = compute_weighted_mean(
                embeddings, torch.tensor(probs, dtype=torch.float32, device=device)
        )        
        diffs_weighted = embeddings - weighted_mean_embeddings.unsqueeze(0)
        weighted_rds_scores = torch.norm(diffs_weighted, p=1, dim=1)
        weighted_rds = (torch.tensor(probs, dtype=torch.float32, device=device) * weighted_rds_scores).sum().item()
        
        # --- EigenEmbed ---
        eigen_embed = compute_eigen_embed(embeddings, alpha=1e-3)
        
        # --- Store norms ---
        norm_dict["EigenEmbed"].append(eigen_embed)
        norm_dict["RDS Score (base)"].append(rds)
        norm_dict["RDS L2 (base)"].append(rds_l2)
        norm_dict["RDS Score (weighted)"].append(weighted_rds)
        
        ### PRO
        norm_dict["PRO"].append(pro_score(gen))
        
        ### Semantic Baselines
        if args.semantic_baselines:
            sem_entropy = compute_semantic_entropy(gen, semantic_model, semantic_tokenizer)
            norm_dict["SE"].append(sem_entropy)

            deg_val, sd_val = compute_deg_semantic_density(gen, semantic_model, semantic_tokenizer)
            norm_dict["Deg"].append(deg_val)
            norm_dict["SD"].append(sd_val)
            
        ### Self-Consistency
        freq = Counter(cleaned_texts)
        major_sample_count = freq.most_common(1)[0][1]
        major_score = major_sample_count / len(cleaned_texts)
        norm_dict['SC'].append(1 - major_score)

    # --- AUROC reporting ---
    results = {}
    print("\n=== Metric Performance (ROC-AUC) ===")
    for method, values in norm_dict.items():
        auc = roc_auc_score(labels, values)
        results[method] = round(auc, 4)
        print(f"{method:45s} → ROC-AUC: {auc:.4f}")
    print("====================================\n")
    
    json_path = f"{config.result_dir}/{args.dataset}__{args.model}__{args.n_samples}__{args.embed_model}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    return results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute uncertainty scores for generated sequences')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--embed_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--semantic_baselines', type=bool, default=True, help='Whether to compute semantic baselines (SE, Deg, SD)')
    args = parser.parse_args()
    
    set_seed(10)
    
    # --- Load semantic model and tokenizer ---
    semantic_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    semantic_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to('cuda')
    
    print(f"Running scoring for {args.dataset} with model {args.model} using {args.embed_model} embeddings, n_samples:{args.n_samples}.")
    main(args, semantic_model, semantic_tokenizer)
