import os
import pickle
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
import config
import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
from sentence_transformers import SentenceTransformer


def approx(probs):
    """Compute PRO score from probabilities."""
    pk = probs[-1]
    score = -np.log(pk) - np.sum([pi * np.log(pi / pk) for pi in probs[:-1]])
    return score


def compute_weighted_mean(embeddings, weights):
    """Compute weighted mean of embeddings."""
    weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(embeddings.device)
    return (weights * embeddings).sum(dim=0)


def compute_metrics(embeddings, mean, weights):
    """
    Compute weighted variance and weighted averages of L1 and L2 norms for embeddings.
    Normalizes results by the sum of weights.
    
    Args:
        embeddings (torch.Tensor): Tensor of shape (n, d) containing n embeddings.
        mean (torch.Tensor): Tensor of shape (d,) representing the weighted mean embedding.
        weights (list or torch.Tensor): Weights for each embedding, shape (n,).
    
    Returns:
        tuple: (weighted_variance, weighted_norms_l1, weighted_norms_l2)
            - weighted_variance (float): Weighted variance (squared L2-norm, normalized).
            - weighted_norms_l1 (float): Weighted average of L1-norms.
            - weighted_norms_l2 (float): Weighted average of L2-norms.
    """
    # Compute differences from the mean
    diffs = embeddings - mean
    
    # Compute norms
    norms_l1 = torch.norm(diffs, p=1, dim=1)  # L1-norm
    norms_l2 = torch.norm(diffs, p=2, dim=1)  # L2-norm
    squared_norms_l2 = torch.sum(diffs ** 2, dim=1)  # Squared L2-norm for variance
    
    # Convert weights to tensor and move to the same device
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(embeddings.device)
    
    # Compute sum of weights for normalization
    sum_weights = weights_tensor.sum()
    
    # Compute weighted sums, normalized by sum of weights
    weighted_variance = (weights_tensor * squared_norms_l2).sum() / sum_weights
    weighted_norms_l1 = (weights_tensor * norms_l1).sum() / sum_weights
    weighted_norms_l2 = (weights_tensor * norms_l2).sum() / sum_weights
    
    return weighted_variance.item(), weighted_norms_l1.item(), weighted_norms_l2.item()


def compute_wasserstein(norms, weights, p=2):
    """Compute generalized p-Wasserstein distance."""
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(norms.device if isinstance(norms, torch.Tensor) else 'cpu')
    return ((weights_tensor * (norms ** p)).sum()) ** (1.0 / p)


def eigen_score(generations, tokenizer=None, model=None, embed_model='all-MiniLM-L6-v2', mode="internal", layer_idx=12):
    """
    Compute EigenScore for a list of generated outputs, supporting both internal LLM hidden states
    and third-party embeddings.
    
    Args:
        generations (list): List of generated text outputs from LLM.
        tokenizer (transformers.AutoTokenizer): Tokenizer for the LLM (required for mode='internal').
        model (transformers.AutoModel): LLM model for hidden states (required for mode='internal').
        embed_model (sentence_transformers.SentenceTransformer): Model for third-party embeddings
            (required for mode='third_party').
        mode (str): 'internal' for LLM hidden states, 'third_party' for external embeddings.
        layer_idx (int): Layer index for hidden states (default: 12).
    
    Returns:
        float: EigenScore (differential entropy from covariance matrix).
    """
    if mode not in ["internal", "third_party"]:
        raise ValueError("mode must be 'internal' or 'third_party'")
    
    embeddings = []
    
    if mode == "internal":
        if tokenizer is None or model is None:
            raise ValueError("tokenizer and model are required for mode='internal'")
        model.eval()
        for g in generations:
            inputs = tokenizer(g, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[layer_idx][:, -1, :].squeeze().cpu().numpy()  # Last token
            embeddings.append(hidden_state)
    
    elif mode == "third_party":
        if embed_model is None:
            raise ValueError("embed_model is required for mode='third_party'")
        embeddings = embed_model.encode(generations, convert_to_numpy=True)
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # Compute covariance matrix
    cov = np.cov(embeddings.T)
    
    # Compute eigenvalues and handle numerical stability
    eigenvalues = np.linalg.eigvals(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Avoid log(0)
    
    # Compute differential entropy (EigenScore)
    eigen_score = np.sum(np.log(eigenvalues))
    
    return eigen_score


def process_generation(generation, model, device):
    """
    Process a single generation dict:
    - compute embeddings
    - weighted mean
    - variance
    - Wasserstein distance
    """
    cleaned_texts = generation["cleaned_generated_texts"]
    samples_avg_nll = generation["samples_avg_nll"]
    probs = np.exp(-np.array(samples_avg_nll))
    probs /= probs.sum()  # normalize
    if probs.sum() == 0:
        print("Warning: probabilities sum to zero, adjusting to uniform.")

    embeddings = model.encode(cleaned_texts, convert_to_tensor=True, device=device)
    
    # Weighted mean
    weighted_mean = compute_weighted_mean(embeddings, probs)
    # Baseline: all weights equal
    baseline_mean = embeddings.mean(dim=0)

    # Variance
    weighted_variance, weighted_norm_l1, weighted_norm_l2 = compute_metrics(embeddings, weighted_mean, probs)
    baseline_variance, baseline_norm_l1, baseline_norm_l2 = compute_metrics(embeddings, baseline_mean, 
                                                                            np.ones(len(probs))/len(probs))

    # Move scalars to CPU Python floats
    # variance = variance.item() if isinstance(variance, torch.Tensor) else float(variance)
    # variance_baseline = variance_baseline.item() if isinstance(variance_baseline, torch.Tensor) else float(variance_baseline)
    # wasserstein_p = wasserstein_p.item() if isinstance(wasserstein_p, torch.Tensor) else float(wasserstein_p)

    return weighted_variance, weighted_norm_l1, weighted_norm_l2, baseline_variance, baseline_norm_l1, baseline_norm_l2


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(args.embed_model).to(device)

    # Load saved generations
    filepath = f'{config.output_dir}/{args.dataset}__{args.model}__generation.pkl'
    with open(filepath, 'rb') as infile:
        generations = pickle.load(infile)

    list_weighted_variance, list_weighted_l1_means, list_weighted_l2_means = [], [], []
    list_baseline_variance, list_baseline_l1_means, list_baseline_l2_means = [], [], []
    nlls, avg_nlls = [], []
    labels = []
    
    for gen in tqdm(generations):
        # Label: 1 = bad, 0 = good
        if args.dataset in ['gsm8k', 'svamp']:
            label = 1 - int(gen[f'eval_score'] == 1.0)
        else:
            label = 1 - (gen[f'eval_score'] > 0.3).astype(int)
        labels.append(label)

        nlls.append(gen["greedy_nll"].item())
        avg_nlls.append(gen["greedy_avg_nll"].item())

        # Process generation
        weighted_variance, weighted_norm_l1, weighted_norm_l2, baseline_variance, baseline_norm_l1, baseline_norm_l2 = process_generation(gen, model, device)
        if any(np.isnan([weighted_variance, weighted_norm_l1, weighted_norm_l2, baseline_variance, baseline_norm_l1, baseline_norm_l2])):
            print(gen)
            print("Warning: NaN encountered in metrics computation.")
        
        list_weighted_variance.append(weighted_variance if not np.isnan(weighted_variance) else 0.0)
        list_weighted_l1_means.append(weighted_norm_l1 if not np.isnan(weighted_norm_l1) else 0.0)
        list_weighted_l2_means.append(weighted_norm_l2 if not np.isnan(weighted_norm_l2) else 0.0)
        list_baseline_variance.append(baseline_variance if not np.isnan(baseline_variance) else 0.0)
        list_baseline_l1_means.append(baseline_norm_l1 if not np.isnan(baseline_norm_l1) else 0.0)
        list_baseline_l2_means.append(baseline_norm_l2 if not np.isnan(baseline_norm_l2) else 0.0)

    # Compute PRO score (optional)
    pro_scores = []
    for gen in generations:
        nll_probs = np.exp(-np.array(gen["samples_nll"]))
        nll_probs /= nll_probs.sum()
        top_probs = np.sort(nll_probs)[::-1]
        alpha = 0.4
        filtered = top_probs[top_probs >= alpha]
        if len(filtered) == 0:
            filtered = top_probs[:1]
        pro_scores.append(approx(filtered))
    pro_scores = np.array(pro_scores)

    # Compute ROC-AUC for all uncertainty metrics
    auc_baseline_var = roc_auc_score(labels, list_baseline_variance)
    auc_weighted_var = roc_auc_score(labels, list_weighted_variance)
    auc_baseline_l1 = roc_auc_score(labels, list_baseline_l1_means)
    auc_weighted_l1 = roc_auc_score(labels, list_weighted_l1_means)
    auc_baseline_l2 = roc_auc_score(labels, list_baseline_l2_means)
    auc_weighted_l2 = roc_auc_score(labels, list_weighted_l2_means)
    auc_nll = roc_auc_score(labels, nlls)
    auc_all = roc_auc_score(labels, avg_nlls)
    auc_pro = roc_auc_score(labels, pro_scores)

    # Optional: report NaNs
    # print(f"Number of NaNs in Variance (W2): {(weighted_variance != weighted_variance).sum()}")
    # print(f"Number of NaNs in Wasserstein L1: {(weighted_norm_l1 != weighted_norm_l1).sum()}")
    # print(f"Number of NaNs in Norm L2: {(weighted_norm_l2 != weighted_norm_l2).sum()}")

    # Prepare report
    from datetime import datetime
    import json
    report = {
        "model": args.model,
        "dataset": args.dataset,
        "n_samples": args.n_samples,
        "run_datetime": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "roc_auc": {
            "baseline_var": round(auc_baseline_var, 4),
            "weighted_var": round(auc_weighted_var, 4),
            "baseline_l1": round(auc_baseline_l1, 4),
            "weighted_l1": round(auc_weighted_l1, 4),
            "baseline_l2": round(auc_baseline_l2, 4),
            "weighted_l2": round(auc_weighted_l2, 4),
            "baseline_all": round(auc_all, 4),
            "baseline_nll": round(auc_nll, 4),
            "pro_score": round(auc_pro, 4),
        }
    }

    # Save JSON
    output_path = os.path.join(config.result_dir, f"{args.dataset}__{args.model}__{args.n_samples}__{args.embed_model}.json")
    os.makedirs(config.result_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Saved report to {output_path}")    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute uncertainty scores for generated sequences')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--embed_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=10)
    args = parser.parse_args()

    main(args)
