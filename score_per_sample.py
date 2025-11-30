import os
import pickle
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
import config
import warnings
warnings.filterwarnings('ignore')
import evaluate

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

from semantic_baselines import compute_clusters
from generation import clean_generation, clean_answer, is_correct

def evaluation_sample(dataset, text, answer, rouge):
    
    if dataset == 'gsm8k':
        text = text.strip()
    else:
        text = clean_generation(text)
        
    if dataset in ['svamp', 'arith']: # exact match for math datasets
        eval_score = text.strip() == answer.strip()
    elif dataset == 'gsm8k':
        model_answer = clean_answer(text)
        eval_score = is_correct(model_answer=model_answer, answer=answer)
    else:
        eval_score = rouge.compute(predictions=[text], references=[answer])['rougeL']
    
    if dataset in ['gsm8k', 'svamp', 'arith']:
        acc = int(eval_score == 1.0)
    else:
        acc = int(eval_score > 0.3)
        
    return acc

from collections import Counter

def main(args):
    experiment_id = os.getpid()
    cache_dir = f"/tmp/rouge_cache_{experiment_id}"
    os.environ['HF_EVALUATE_CACHE'] = cache_dir
    rouge = evaluate.load('rouge', experiment_id=experiment_id, cache_dir=cache_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer(args.embed_model).to(device)

    # --- Load generations ---
    gen_path = f"{config.output_dir}/{args.dataset}__{args.model}__generation.pkl"
    with open(gen_path, "rb") as infile:
        generations = pickle.load(infile)

    accuracy = {}
    
    for i, gen in enumerate(tqdm(generations, desc="Processing generations")):
        
        # --- Find the least uncertain samples ---
        cleaned_texts = gen["cleaned_generated_texts"]
        samples_avg_nll = gen["samples_avg_nll"]
        samples_nll = gen["samples_nll"]
        
        # --- OT score ---
        embeddings = embed_model.encode(cleaned_texts, convert_to_tensor=True, device=device)
        mean_embedding = torch.mean(embeddings, dim=0)
        diffs = embeddings - mean_embedding 
        ot_scores = torch.norm(diffs, p=1, dim=1)
        
        probs = np.exp(-np.array(samples_avg_nll))
        probs /= probs.sum() 
        weighted_mean_embeddings = compute_weighted_mean(
                embeddings, torch.tensor(probs, dtype=torch.float32, device=device)
        )
        
        diffs_weighted = embeddings - weighted_mean_embeddings.unsqueeze(0)
        weighted_ot_scores = torch.norm(diffs_weighted, p=1, dim=1)
        
        # --- Penalized Weighted OT Score ---
        probs_t = torch.tensor(probs, dtype=torch.float32, device=device)
        eps = 1e-12
        penal_weighted_ot_scores = weighted_ot_scores / (probs_t + eps)
        
        # --- Ranking and find samples ---
        nll_sample = cleaned_texts[np.argmin(samples_nll)]
        avg_nll_sample = cleaned_texts[np.argmin(samples_avg_nll)]
        ot_sample = cleaned_texts[torch.argmin(ot_scores).item()]
        weighted_ot_sample = cleaned_texts[torch.argmin(weighted_ot_scores).item()]
        penal_weighted_ot_sample = cleaned_texts[torch.argmin(penal_weighted_ot_scores).item()]

        freq = Counter(cleaned_texts)
        majority_sample = freq.most_common(1)[0][0]

        # --- Evaluation ---
        for method, sample in zip(
            ["nll", "avg_nll", "ot", "weighted_ot", "penal_weighted_ot", "majority"],
            [nll_sample, avg_nll_sample, ot_sample, weighted_ot_sample, penal_weighted_ot_sample, majority_sample]
        ):
            acc = evaluation_sample(
                dataset=args.dataset,
                text=sample,
                answer=gen["answer"],
                rouge=rouge
            )
            if method not in accuracy:
                accuracy[method] = []
                
            accuracy[method].append(acc)
            
            # For debug
            if i < 3:
                print(f"Sample {i} | Method: {method} | Acc: {acc} | Sample: {sample[:50]}...")
    
    # --- Reporting ---
    results = {}
    print("\n=== Metric Performance ===")
    for method, values in accuracy.items():
        final_acc = np.mean(values)
        results[method] = final_acc
        print(f"{method:55s} → ACC: {final_acc:.4f}")

    # --- Save results ---
    json_path = f"{config.result_dir}/{args.dataset}__{args.model}__sample__{args.version}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"📄 Saved AUROC summary to {json_path}")

    return results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute uncertainty scores for generated sequences')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--embed_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--version', type=str, default='20251129', help='Version identifier for the run')
    args = parser.parse_args()

    main(args)