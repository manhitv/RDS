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
from generation import MODEL_PATH_DICT
from semantic_baselines import compute_semantic_entropy, compute_deg_semantic_density


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


def eigen_score(prompt, generations, tokenizer=None, model=None, sentence_embeddings=None, mode="internal"):
    """
    Compute EigenScore for a list of generated outputs, supporting both internal LLM hidden states
    and third-party embeddings.
    """
    alpha = 1e-3
    embeddings = []
    
    if mode == "internal":
        if tokenizer is None or model is None:
            raise ValueError("tokenizer and model are required for mode='internal'")
        model.eval()
        for output in generations:
            # Encode
            full_text = prompt + output
            inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # list[layer][batch, seq, hidden]
            
            # Pick middle layer by default
            layer_idx = len(hidden_states) // 2

            # Use token before the last one (num_tokens - 2)
            num_tokens = inputs["input_ids"].shape[1]
            token_idx = max(num_tokens - 2, 0) # based on author's implementation
            
            emb = hidden_states[layer_idx][:, token_idx, :].squeeze().cpu().numpy()
            embeddings.append(emb)
    
    elif mode == "third_party":
        if sentence_embeddings is None:
            raise ValueError("embeddings are required for mode='third_party'")
        embeddings = sentence_embeddings.cpu().numpy() if isinstance(sentence_embeddings, torch.Tensor) else sentence_embeddings
    
    # Compute covariance matrix
    embeddings = np.array(embeddings)
    cov = np.cov(embeddings.T) + alpha * np.eye(embeddings.shape[1])

    # SVD and eigen score
    u, s, vT = np.linalg.svd(cov)
    eigen_score = np.mean(np.log10(s))
    
    return eigen_score


# ---------------- 
### PRO Score
# ----------------
def approx(probs):
    """Compute PRO score from probabilities."""
    pk = probs[-1]
    score = -np.log(pk) - np.sum([pi * np.log(pi / pk) for pi in probs[:-1]])
    return score


def pro_score(generation, alpha=0.4):
    """Compute PRO score from generation."""
    nll_probs = np.exp(-np.array(generation["samples_nll"]))
    nll_probs /= nll_probs.sum()
    top_probs = np.sort(nll_probs)[::-1]
    filtered = top_probs[top_probs >= alpha]
    if len(filtered) == 0:
        filtered = top_probs[:1]
        
    return approx(filtered)

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
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer(args.embed_model).to(device)

    # Load saved generations
    filepath = f'{config.output_dir}/{args.dataset}__{args.model}__generation.pkl'
    with open(filepath, 'rb') as infile:
        generations = pickle.load(infile)

    list_weighted_variance, list_weighted_l1_means, list_weighted_l2_means = [], [], []
    list_baseline_variance, list_baseline_l1_means, list_baseline_l2_means = [], [], []
    
    # Baselines: NLL and avg NLL
    nlls, avg_nlls = [], []

    # PRO scores
    pro_scores = []

    # Other baselines
    eigen_scores_llm, eigen_scores_embed = [], []
    semantic_entropy, semantic_density, deg = [], [], []
    
    if args.eigen_baselines:
        model, tokenizer = load_model_from_path(args.model, device)
    
    if args.semantic_baselines:
        semantic_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        semantic_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)

    labels = []
    
    for gen in tqdm(generations):
        # Label: 1 = bad, 0 = good
        if args.dataset in ['gsm8k', 'svamp']:
            label = 1 - int(gen[f'eval_score'] == 1.0)
        else:
            label = 1 - (gen[f'eval_score'] > 0.3).astype(int)
        labels.append(label)

        ### NLL baselines
        nlls.append(gen["greedy_nll"].item())
        avg_nlls.append(gen["greedy_avg_nll"].item())

        ### Samples processing
        prompt = gen["prompt"]
        cleaned_texts = gen["cleaned_generated_texts"]
        samples_avg_nll = gen["samples_avg_nll"]
        
        probs = np.exp(-np.array(samples_avg_nll))
        probs /= probs.sum()  # normalize
        if probs.sum() == 0:
            print("Warning: probabilities sum to zero, adjusting to uniform.")

        embeddings = embed_model.encode(cleaned_texts, convert_to_tensor=True, device=device)
        
        ### VAR computations
        # Weighted mean
        weighted_mean = compute_weighted_mean(embeddings, probs)
        # Baseline: all weights equal
        baseline_mean = embeddings.mean(dim=0)

        weighted_variance, weighted_norm_l1, weighted_norm_l2 = compute_metrics(embeddings, weighted_mean, probs)
        baseline_variance, baseline_norm_l1, baseline_norm_l2 = compute_metrics(embeddings, baseline_mean, 
                                                                                np.ones(len(probs))/len(probs))

        if any(np.isnan([weighted_variance, weighted_norm_l1, weighted_norm_l2, baseline_variance, baseline_norm_l1, baseline_norm_l2])):
            print(gen)
            print("Warning: NaN encountered in metrics computation.")
        
        list_weighted_variance.append(weighted_variance)
        list_weighted_l1_means.append(weighted_norm_l1)
        list_weighted_l2_means.append(weighted_norm_l2)
        list_baseline_variance.append(baseline_variance)
        list_baseline_l1_means.append(baseline_norm_l1)
        list_baseline_l2_means.append(baseline_norm_l2)

        ### PRO score
        pro_scores.append(pro_score(gen))
        
        
        if args.eigen_baselines:
            ### EigenScore
            eigen_embed = eigen_score(prompt=prompt, generations=cleaned_texts, sentence_embeddings=embeddings, mode="third_party")
            eigen_scores_embed.append(eigen_embed)

            eigen_llm = eigen_score(prompt=prompt, generations=cleaned_texts, tokenizer=tokenizer, model=model, mode="internal")
            eigen_scores_llm.append(eigen_llm)
        
        if args.semantic_baselines:    
            ### Semantic Entropy
            sem_entropy = compute_semantic_entropy(gen, semantic_model, semantic_tokenizer)
            semantic_entropy.append(sem_entropy)

            ### Deg & Semantic Density
            deg_val, sd_val = compute_deg_semantic_density(gen, semantic_model, semantic_tokenizer)
            deg.append(deg_val)
            semantic_density.append(sd_val)
            
    # Compute ROC-AUC for all uncertainty metrics
    auc_baseline_var = roc_auc_score(labels, list_baseline_variance)
    auc_weighted_var = roc_auc_score(labels, list_weighted_variance)
    auc_baseline_l1 = roc_auc_score(labels, list_baseline_l1_means)
    auc_weighted_l1 = roc_auc_score(labels, list_weighted_l1_means)
    auc_baseline_l2 = roc_auc_score(labels, list_baseline_l2_means)
    auc_weighted_l2 = roc_auc_score(labels, list_weighted_l2_means)
    
    # AUROC Baselines
    auc_nll = roc_auc_score(labels, nlls)
    auc_all = roc_auc_score(labels, avg_nlls)
    auc_pro = roc_auc_score(labels, pro_scores)

    if args.eigen_baselines:
        auc_eigen_llm = roc_auc_score(labels, eigen_scores_llm)
        auc_eigen_embed = roc_auc_score(labels, eigen_scores_embed)
        
    if args.semantic_baselines:    
        auc_semantic_entropy = roc_auc_score(labels, semantic_entropy)
        auc_deg = roc_auc_score(labels, deg)
        auc_semantic_density = roc_auc_score(labels, semantic_density)

    # Prepare report
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
            "eigen_llm": round(auc_eigen_llm, 4) if args.eigen_baselines else None,
            "eigen_embed": round(auc_eigen_embed, 4) if args.eigen_baselines else None,
            "semantic_entropy": round(auc_semantic_entropy, 4) if args.semantic_baselines else None,
            "deg": round(auc_deg, 4) if args.semantic_baselines else None,
            "semantic_density": round(auc_semantic_density, 4) if args.semantic_baselines else None,
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
    parser.add_argument('--eigen_baselines', type=bool, default=False, help='Whether to compute eigen baselines')
    parser.add_argument('--semantic_baselines', type=bool, default=False, help='Whether to compute semantic baselines')
    args = parser.parse_args()

    main(args)
