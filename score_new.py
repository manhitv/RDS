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


def compute_metrics(embeddings, mean, weights):
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

    norms = torch.norm(diffs, p=1, dim=1)  # squared norms for variance
    weighted = (weights_tensor * norms).sum()
    return weighted.item()

def compute_metrics_origin(embeddings, mean, weights, p=1):
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

# def compute_metrics(embeddings, mean, weights):
#     """
#     Compute weighted variance and weighted averages of L1 and L2 norms for embeddings.
#     Normalizes results by the sum of weights.
    
#     Args:
#         embeddings (torch.Tensor): Tensor of shape (n, d) containing n embeddings.
#         mean (torch.Tensor): Tensor of shape (d,) representing the weighted mean embedding.
#         weights (list or torch.Tensor): Weights for each embedding, shape (n,).
    
#     Returns:
#         tuple: (weighted_variance, weighted_norms_l1, weighted_norms_l2)
#             - weighted_variance (float): Weighted variance (squared L2-norm, normalized).
#             - weighted_norms_l1 (float): Weighted average of L1-norms.
#             - weighted_norms_l2 (float): Weighted average of L2-norms.
#     """
#     # Compute differences from the mean
#     diffs = embeddings - mean
    
#     # Compute norms
#     norms_l1 = torch.norm(diffs, p=1, dim=1)  # L1-norm
#     norms_l2 = torch.norm(diffs, p=2, dim=1)  # L2-norm
#     squared_norms_l2 = torch.sum(diffs ** 2, dim=1)  # Squared L2-norm for variance
    
#     # Convert weights to tensor and move to the same device
#     weights_tensor = torch.tensor(weights, dtype=torch.float32).to(embeddings.device)
    
#     # Compute sum of weights for normalization
#     sum_weights = weights_tensor.sum()
    
#     # Compute weighted sums, normalized by sum of weights
#     weighted_variance = (weights_tensor * squared_norms_l2).sum() / sum_weights
#     weighted_norms_l1 = (weights_tensor * norms_l1).sum() / sum_weights
#     weighted_norms_l2 = (weights_tensor * norms_l2).sum() / sum_weights
    
#     return weighted_variance.item(), weighted_norms_l1.item(), weighted_norms_l2.item()


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


# ----------------
### Main function
# ----------------
# def main(args):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     embed_model = SentenceTransformer(args.embed_model).to(device)

#     # Load saved generations
#     filepath = f'{config.output_dir}/{args.dataset}__{args.model}__generation.pkl'
#     with open(filepath, 'rb') as infile:
#         generations = pickle.load(infile)

#     list_weighted_variance, list_weighted_l1_means, list_weighted_l2_means = [], [], []
#     list_baseline_variance, list_baseline_l1_means, list_baseline_l2_means = [], [], []
    
#     # Baselines: NLL and avg NLL
#     nlls, avg_nlls = [], []

#     # PRO scores
#     pro_scores = []

#     # Other baselines
#     eigen_scores_llm, eigen_scores_embed = [], []
#     semantic_entropy, semantic_density, deg = [], [], []
    
#     if args.eigen_baselines:
#         model, tokenizer = load_model_from_path(args.model, device)
    
#     if args.semantic_baselines:
#         semantic_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
#         semantic_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)

#     labels = []
    
#     for gen in tqdm(generations):
#         # Label: 1 = bad, 0 = good
#         if args.dataset in ['gsm8k', 'svamp']:
#             label = 1 - int(gen[f'eval_score'] == 1.0)
#         else:
#             label = 1 - (gen[f'eval_score'] > 0.3).astype(int)
#         labels.append(label)

#         ### NLL baselines
#         nlls.append(gen["greedy_nll"].item())
#         avg_nlls.append(gen["greedy_avg_nll"].item())

#         ### Samples processing
#         prompt = gen["prompt"]
#         cleaned_texts = gen["cleaned_generated_texts"]
        
#         if args.prob_score == 'nll':
#             samples_avg_nll = gen["samples_nll"]
#         else:
#             samples_avg_nll = gen["samples_avg_nll"] # might change to samples_nll depending on use case
        
#         probs = np.exp(-np.array(samples_avg_nll))
#         probs /= probs.sum()  # normalize
#         if probs.sum() == 0:
#             print("Warning: probabilities sum to zero, adjusting to uniform.")

#         embeddings = embed_model.encode(cleaned_texts, convert_to_tensor=True, device=device)
        
#         ### VAR computations
#         # Weighted mean
#         weighted_mean = compute_weighted_mean(embeddings, probs)
#         # Baseline: all weights equal
#         baseline_mean = embeddings.mean(dim=0)

#         weighted_variance, weighted_norm_l1, weighted_norm_l2 = compute_metrics(embeddings, weighted_mean, probs)
#         baseline_variance, baseline_norm_l1, baseline_norm_l2 = compute_metrics(embeddings, baseline_mean, 
#                                                                                 np.ones(len(probs))/len(probs))

#         if any(np.isnan([weighted_variance, weighted_norm_l1, weighted_norm_l2, baseline_variance, baseline_norm_l1, baseline_norm_l2])):
#             print(gen)
#             print("Warning: NaN encountered in metrics computation.")
        
#         list_weighted_variance.append(weighted_variance)
#         list_weighted_l1_means.append(weighted_norm_l1)
#         list_weighted_l2_means.append(weighted_norm_l2)
#         list_baseline_variance.append(baseline_variance)
#         list_baseline_l1_means.append(baseline_norm_l1)
#         list_baseline_l2_means.append(baseline_norm_l2)

#         ### PRO score
#         pro_scores.append(pro_score(gen))
        
#         if args.eigen_embed_only:
#             eigen_embed = eigen_score_origin_v2(prompt=prompt, sentence_embeddings=embeddings, mode="third_party")
#             eigen_scores_embed.append(eigen_embed)
        
#         if args.eigen_baselines:
#             ### EigenScore
#             if not args.model.startswith("gemma"):
#                 eigen_llm = eigen_score_origin_v2(prompt=prompt, tokenizer=tokenizer, model=model, mode="internal", n_samples=args.n_samples)
#                 # eigen_llm = eigen_score(prompt=prompt, generations=cleaned_texts, tokenizer=tokenizer, model=model, mode="internal")
#                 eigen_scores_llm.append(eigen_llm)
        
#         if args.semantic_baselines:    
#             ### Semantic Entropy
#             sem_entropy = compute_semantic_entropy(gen, semantic_model, semantic_tokenizer)
#             semantic_entropy.append(sem_entropy)

#             ### Deg & Semantic Density
#             deg_val, sd_val = compute_deg_semantic_density(gen, semantic_model, semantic_tokenizer)
#             deg.append(deg_val)
#             semantic_density.append(sd_val)
            
#     # Compute ROC-AUC for all uncertainty metrics
#     auc_baseline_var = roc_auc_score(labels, list_baseline_variance)
#     auc_weighted_var = roc_auc_score(labels, list_weighted_variance)
#     auc_baseline_l1 = roc_auc_score(labels, list_baseline_l1_means)
#     auc_weighted_l1 = roc_auc_score(labels, list_weighted_l1_means)
#     auc_baseline_l2 = roc_auc_score(labels, list_baseline_l2_means)
#     auc_weighted_l2 = roc_auc_score(labels, list_weighted_l2_means)
    
#     # AUROC Baselines
#     auc_nll = roc_auc_score(labels, nlls)
#     auc_all = roc_auc_score(labels, avg_nlls)
#     auc_pro = roc_auc_score(labels, pro_scores)

#     if args.eigen_baselines:
#         if not args.model.startswith("gemma"): # Gemma models do not support LLM-based EigenScore
#             auc_eigen_llm = roc_auc_score(labels, eigen_scores_llm)
#         else:
#             auc_eigen_llm = None
    
#     if args.eigen_embed_only:    
#         auc_eigen_embed = roc_auc_score(labels, eigen_scores_embed)
        
#     if args.semantic_baselines:    
#         auc_semantic_entropy = roc_auc_score(labels, semantic_entropy)
#         auc_deg = roc_auc_score(labels, deg)
#         auc_semantic_density = roc_auc_score(labels, semantic_density)

#     # Save raw results for further analysis
#     results = {
#         "labels": labels,
#         "list_baseline_variance": list_baseline_variance,
#         "list_weighted_variance": list_weighted_variance,
#         "list_baseline_l1_means": list_baseline_l1_means,
#         "list_weighted_l1_means": list_weighted_l1_means,
#         "list_baseline_l2_means": list_baseline_l2_means,
#         "list_weighted_l2_means": list_weighted_l2_means,
#         "nlls": nlls,
#         "avg_nlls": avg_nlls,
#         "pro_scores": pro_scores,
#         "eigen_scores_llm": eigen_scores_llm if args.eigen_baselines else None,
#         "eigen_scores_embed": eigen_scores_embed if args.eigen_embed_only else None,
#         "semantic_entropy": semantic_entropy if args.semantic_baselines else None,
#         "deg": deg if args.semantic_baselines else None,
#         "semantic_density": semantic_density if args.semantic_baselines else None,
#     }

#     output_path = os.path.join(config.output_dir, f"{args.dataset}__{args.model}__{args.n_samples}__{args.prob_score}__{args.embed_model}_raw_scores.pkl")
#     os.makedirs(config.output_dir, exist_ok=True)
#     with open(output_path, "wb") as f:
#         pickle.dump(results, f)

#     print(f"Saved report to {output_path}")  
    
#     # Prepare report
#     report = {
#         "model": args.model,
#         "dataset": args.dataset,
#         "n_samples": args.n_samples,
#         "embed_model": args.embed_model,
#         "prob_score": args.prob_score,
#         "version": args.version,
#         "run_datetime": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
#         "roc_auc": {
#             "baseline_var": round(auc_baseline_var, 4),
#             "weighted_var": round(auc_weighted_var, 4),
#             "baseline_l1": round(auc_baseline_l1, 4),
#             "weighted_l1": round(auc_weighted_l1, 4),
#             "baseline_l2": round(auc_baseline_l2, 4),
#             "weighted_l2": round(auc_weighted_l2, 4),
#             "baseline_all": round(auc_all, 4),
#             "baseline_nll": round(auc_nll, 4),
#             "pro_score": round(auc_pro, 4),
#             "eigen_llm": round(auc_eigen_llm, 4) if (args.eigen_baselines and auc_eigen_llm is not None) else None,
#             "eigen_embed": round(auc_eigen_embed, 4) if (args.eigen_embed_only and auc_eigen_embed is not None) else None,
#             "semantic_entropy": round(auc_semantic_entropy, 4) if args.semantic_baselines else None,
#             "deg": round(auc_deg, 4) if args.semantic_baselines else None,
#             "semantic_density": round(auc_semantic_density, 4) if args.semantic_baselines else None,
#         }
#     }

#     # Save JSON
#     output_path = os.path.join(config.result_dir, f"{args.dataset}__{args.model}__{args.n_samples}__{args.prob_score}__{args.embed_model}__{args.version}.json")
#     os.makedirs(config.result_dir, exist_ok=True)
#     with open(output_path, "w") as f:
#         json.dump(report, f, indent=4)

#     print(f"Saved report to {output_path}")    


#### -------------------- 20251105
# def main(args, semantic_model, semantic_tokenizer):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     embed_model = SentenceTransformer(args.embed_model).to(device)

#     # --- Load generations ---
#     gen_path = f"{config.output_dir}/{args.dataset}__{args.model}__generation.pkl"
#     with open(gen_path, "rb") as infile:
#         generations = pickle.load(infile)

#     # --- Cluster cache paths ---
#     cluster_cache_path = f"{config.output_dir}/{args.dataset}__{args.model}__clusters.pkl"
#     clusters_cached = os.path.exists(cluster_cache_path)

#     if clusters_cached:
#         print(f"✅ Loading existing clusters from: {cluster_cache_path}")
#         with open(cluster_cache_path, "rb") as f:
#             cluster_cache = pickle.load(f)
#     else:
#         print(f"⚙️  Computing semantic clusters (first run)...")
#         cluster_cache = {}

#     labels = []
#     norm_dict = defaultdict(list)

#     ### Semantic baselines storage
#     semantic_entropy = []
#     deg = []
#     semantic_density = []
#     for i, gen in enumerate(tqdm(generations, desc="Processing generations")):
#         # --- Label ---
#         if args.dataset in ['gsm8k', 'svamp', 'arith']:
#             label = 1 - int(gen['eval_score'] == 1.0)
#         else:
#             label = 1 - int(gen['eval_score'] > 0.3)
#         labels.append(label)

#         cleaned_texts = gen["cleaned_generated_texts"]
#         samples_avg_nll = gen["samples_avg_nll"]

#         # --- Weighting sets ---
#         probs = np.exp(-np.array(samples_avg_nll))
#         probs /= probs.sum()
#         uniform_probs = np.ones(len(probs)) / len(probs)

#         # --- Embeddings ---
#         embeddings = embed_model.encode(cleaned_texts, convert_to_tensor=True, device=device)

#         # --- Load or compute clusters ---
#         if clusters_cached and i in cluster_cache:
#             cluster_ids = cluster_cache[i]["cluster_ids"]
#         else:
#             cluster_ids = compute_semantic_similarity(gen, semantic_model, semantic_tokenizer, device=device)
#             cluster_cache[i] = {"cluster_ids": cluster_ids}

#         # --- Build cluster info ---
#         clusters = defaultdict(list)
#         cluster_weights_prob = defaultdict(list)
#         cluster_weights_uniform = defaultdict(list)

#         for emb, w_prob, cid in zip(embeddings, probs, cluster_ids):
#             clusters[cid].append(emb)
#             cluster_weights_prob[cid].append(w_prob)
#             cluster_weights_uniform[cid].append(1.0 / len(probs))

#         cluster_info = {}
#         for cid, embs_list in clusters.items():
#             embs_tensor = torch.stack(embs_list)
#             cluster_info[cid] = {
#                 "embs": embs_tensor,
#                 "size": len(embs_list),
#                 "prob_sum": np.sum(cluster_weights_prob[cid]),
#                 "weights_prob": np.array(cluster_weights_prob[cid]),
#                 "weights_uniform": np.array(cluster_weights_uniform[cid]),
#             }

#         # --- Base metrics ---
#         for weight_mode, w in [('prob', probs), ('uniform', uniform_probs)]:
#             baseline_mean = compute_weighted_mean(
#                 embeddings, torch.tensor(w, dtype=torch.float32, device=device)
#             )
#             lp_base = compute_metrics(embeddings, baseline_mean, w, p=1)
#             eigen_base = compute_eigen_embed(embeddings, alpha=1e-3)
#             norm_dict[f"lp_base_{weight_mode}"].append(lp_base)
#             norm_dict[f"eigen_embed_base_{weight_mode}"].append(eigen_base)

#         # --- Cluster-level metrics ---
#         for weight_type in ['prob', 'uniform']:
#             cluster_norms, cluster_eigens, cluster_sizes, cluster_prob_sums = [], [], [], []
#             for cid, info in cluster_info.items():
#                 embs_tensor = info["embs"]
#                 weights = torch.tensor(info[f"weights_{weight_type}"], dtype=torch.float32, device=device)
#                 cluster_mean = compute_weighted_mean(embs_tensor, weights)

#                 cluster_norm = compute_metrics(embs_tensor, cluster_mean, weights.cpu().numpy(), p=1)
#                 cluster_eigen = compute_eigen_embed(embs_tensor, alpha=1e-3)

#                 cluster_norms.append(cluster_norm)
#                 cluster_eigens.append(cluster_eigen)
#                 cluster_sizes.append(info["size"])
#                 cluster_prob_sums.append(info["prob_sum"])

#             cluster_sizes = np.array(cluster_sizes)
#             cluster_prob_sums = np.array(cluster_prob_sums)
#             size_weights = cluster_sizes / cluster_sizes.sum()
#             prob_weights = cluster_prob_sums / cluster_prob_sums.sum()
#             uniform_cluster_weights = np.ones_like(size_weights) / len(size_weights)

#             for mode, w_cluster in [
#                 ('size', size_weights),
#                 ('prob', prob_weights),
#                 ('uniform', uniform_cluster_weights)
#             ]:
#                 lp_cluster = np.average(cluster_norms, weights=w_cluster)
#                 eig_cluster = np.average(cluster_eigens, weights=w_cluster)
#                 norm_dict[f"lp_cluster_{weight_type}_{mode}"].append(lp_cluster)
#                 norm_dict[f"eigen_embed_cluster_{weight_type}_{mode}"].append(eig_cluster)

#         if args.semantic_baselines:    
#             ### Semantic Entropy
#             sem_entropy = compute_semantic_entropy(gen, semantic_model, semantic_tokenizer)
#             semantic_entropy.append(sem_entropy)

#             ### Deg & Semantic Density
#             deg_val, sd_val = compute_deg_semantic_density(gen, semantic_model, semantic_tokenizer)
#             deg.append(deg_val)
#             semantic_density.append(sd_val)
    
#     if args.semantic_baselines:    
#         norm_dict["semantic_entropy"] = semantic_entropy
#         norm_dict["deg"] = deg
#         norm_dict["semantic_density"] = semantic_density

#     # --- Save computed clusters (only first run) ---
#     if not clusters_cached:
#         with open(cluster_cache_path, "wb") as f:
#             pickle.dump(cluster_cache, f)
#         print(f"💾 Saved new cluster assignments to {cluster_cache_path}")

#     # --- AUROC reporting ---
#     results = {}
#     print("\n=== Metric Performance (ROC-AUC) ===")
#     for method, values in norm_dict.items():
#         auc = roc_auc_score(labels, values)
#         results[method] = round(auc, 4)
#         print(f"{method:45s} → ROC-AUC: {auc:.4f}")
#     print("====================================\n")
    
#     json_path = f"{config.result_dir}/{args.dataset}__{args.model}__{args.version}.json"
#     with open(json_path, "w") as f:
#         json.dump(results, f, indent=4)
# #     print(f"📄 Saved AUROC summary to {json_path}")

#     return results

from semantic_baselines import compute_clusters

### 20251106 - NEW CLUSTERING METHOD
def main(args, semantic_model=None, semantic_tokenizer=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer(args.embed_model).to(device)

    # --- Load generations ---
    gen_path = f"{config.output_dir}/{args.dataset}__{args.model}__generation.pkl"
    with open(gen_path, "rb") as infile:
        generations = pickle.load(infile)

    # --- Cluster cache paths ---
    cluster_cache_path = f"{config.output_dir}/{args.dataset}__{args.model}__{args.cluster_mode}__clusters.pkl"
    clusters_cached = os.path.exists(cluster_cache_path)

    if clusters_cached:
        print(f"✅ Loading existing clusters from: {cluster_cache_path}")
        with open(cluster_cache_path, "rb") as f:
            cluster_cache = pickle.load(f)
    else:
        print(f"⚙️  Computing clusters (first run)...")
        cluster_cache = {}

    labels = []
    norm_dict = defaultdict(list)
    semantic_entropy, deg, semantic_density = [], [], []
    
    for i, gen in enumerate(tqdm(generations, desc="Processing generations")):
        # --- Label ---
        if args.dataset in ['gsm8k', 'svamp', 'arith']:
            label = 1 - int(gen['eval_score'] == 1.0)
        else:
            label = 1 - int(gen['eval_score'] > 0.3)
        labels.append(label)

        cleaned_texts = gen["cleaned_generated_texts"]
        samples_avg_nll = gen["samples_avg_nll"]
        
        norm_dict['nll'].append(gen["greedy_nll"].item())
        norm_dict['avg_nll'].append(gen["greedy_avg_nll"].item())
        norm_dict['pro'].append(pro_score(gen))
        
        # --- Weighting sets ---
        probs = np.exp(-np.array(samples_avg_nll))
        probs /= probs.sum()
        uniform_probs = np.ones(len(probs)) / len(probs)

        # --- Embeddings ---
        embeddings = embed_model.encode(cleaned_texts, convert_to_tensor=True, device=device)

        # --- Load or compute clusters ---
        if clusters_cached and i in cluster_cache:
            cluster_variants = cluster_cache[i]["cluster_variants"]
        else:
            if args.cluster_mode == 'semantic_clustering':
                cluster_variants = compute_clusters(
                    gen,
                    embed_model,
                    device=device,
                    cluster_mode=args.cluster_mode,  # "pca_only", "kmeans_only", "pca_then_kmeans"
                    max_components=10,
                    max_k=10,
                    semantic_model=semantic_model,
                    semantic_tokenizer=semantic_tokenizer
                )
            else:
                cluster_variants = compute_clusters(
                    gen,
                    embed_model,
                    device=device,
                    cluster_mode=args.cluster_mode,  # "pca_only", "kmeans_only", "pca_then_kmeans"
                    max_components=10,
                    max_k=10
                )
            cluster_cache[i] = {"cluster_variants": cluster_variants}

        # --- Base metrics ---
        for weight_mode, w in [('prob', probs), ('uniform', uniform_probs)]:
            baseline_mean = compute_weighted_mean(
                embeddings, torch.tensor(w, dtype=torch.float32, device=device)
            )
            lp_base = compute_metrics(embeddings, baseline_mean, w)
            eigen_base = compute_eigen_embed(embeddings, alpha=1e-3)
            norm_dict[f"lp_base_{weight_mode}"].append(lp_base)
            norm_dict[f"eigen_embed_base_{weight_mode}"].append(eigen_base)

        # --- Cluster-level metrics (per PCA/KMeans configuration) ---
        for key, cluster_ids in cluster_variants.items():
            clusters = defaultdict(list)
            cluster_weights_prob = defaultdict(list)
            cluster_weights_uniform = defaultdict(list)

            for emb, w_prob, cid in zip(embeddings, probs, cluster_ids):
                clusters[cid].append(emb)
                cluster_weights_prob[cid].append(w_prob)
                cluster_weights_uniform[cid].append(1.0 / len(probs))

            cluster_info = {}
            for cid, embs_list in clusters.items():
                embs_tensor = torch.stack(embs_list)
                cluster_info[cid] = {
                    "embs": embs_tensor,
                    "size": len(embs_list),
                    "prob_sum": np.sum(cluster_weights_prob[cid]),
                    "weights_prob": np.array(cluster_weights_prob[cid]),
                    "weights_uniform": np.array(cluster_weights_uniform[cid]),
                }

            for weight_type in ['prob', 'uniform']:
                cluster_norms, cluster_eigens, cluster_sizes, cluster_prob_sums = [], [], [], []
                for cid, info in cluster_info.items():
                    embs_tensor = info["embs"]
                    weights = torch.tensor(info[f"weights_{weight_type}"], dtype=torch.float32, device=device)
                    cluster_mean = compute_weighted_mean(embs_tensor, weights)
                    cluster_norm = compute_metrics(embs_tensor, cluster_mean, weights.cpu().numpy())
                    cluster_eigen = compute_eigen_embed(embs_tensor, alpha=1e-3)
                    cluster_norms.append(cluster_norm)
                    cluster_eigens.append(cluster_eigen)
                    cluster_sizes.append(info["size"])
                    cluster_prob_sums.append(info["prob_sum"])

                cluster_sizes = np.array(cluster_sizes)
                cluster_prob_sums = np.array(cluster_prob_sums)
                size_weights = cluster_sizes / cluster_sizes.sum()
                prob_weights = cluster_prob_sums / cluster_prob_sums.sum()
                uniform_cluster_weights = np.ones_like(size_weights) / len(size_weights)

                for mode, w_cluster in [
                    ('size', size_weights),
                    ('prob', prob_weights),
                    ('uniform', uniform_cluster_weights)
                ]:
                    lp_cluster = np.average(cluster_norms, weights=w_cluster)
                    eig_cluster = np.average(cluster_eigens, weights=w_cluster)
                    prefix = f"{key}_"
                    norm_dict[f"{prefix}lp_cluster_{weight_type}_{mode}"].append(lp_cluster)
                    norm_dict[f"{prefix}eigen_embed_cluster_{weight_type}_{mode}"].append(eig_cluster)

        # --- Optional semantic baselines ---
        if args.semantic_baselines:
            sem_entropy = compute_semantic_entropy(gen, semantic_model, semantic_tokenizer)
            semantic_entropy.append(sem_entropy)

            deg_val, sd_val = compute_deg_semantic_density(gen, semantic_model, semantic_tokenizer)
            deg.append(deg_val)
            semantic_density.append(sd_val)

    if args.semantic_baselines:
        norm_dict["semantic_entropy"] = semantic_entropy
        norm_dict["deg"] = deg
        norm_dict["semantic_density"] = semantic_density

    # --- Save new clusters if not cached ---
    if not clusters_cached:
        with open(cluster_cache_path, "wb") as f:
            pickle.dump(cluster_cache, f)
        print(f"💾 Saved new cluster assignments to {cluster_cache_path}")

    # --- AUROC reporting ---
    results = {}
    print("\n=== Metric Performance (ROC-AUC) ===")
    for method, values in norm_dict.items():
        try:
            auc = roc_auc_score(labels, values)
            results[method] = round(auc, 4)
            print(f"{method:55s} → ROC-AUC: {auc:.4f}")
        except Exception:
            continue
    print("====================================\n")

    # --- Select best-performing metric ---
    best_metric = max(results.items(), key=lambda x: x[1])
    results["best_config"] = {"name": best_metric[0], "auc": best_metric[1]}
    print(f"🏆 Best configuration: {best_metric[0]} → {best_metric[1]:.4f}")

    # --- Save results ---
    json_path = f"{config.result_dir}/{args.dataset}__{args.model}__{args.cluster_mode}__{args.version}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"📄 Saved AUROC summary to {json_path}")

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
    parser.add_argument('--version', type=str, default='20251106', help='Version identifier for the run')
    parser.add_argument('--semantic_baselines', type=bool, default=False, help='Whether to compute semantic baselines')
    parser.add_argument('--cluster_mode', type=str, default='pca_then_kmeans', help='Clustering mode: pca_only, kmeans_only, pca_then_kmeans')
    args = parser.parse_args()

    semantic_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    semantic_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to('cuda')
    main(args, semantic_model, semantic_tokenizer)
