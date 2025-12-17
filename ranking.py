
import warnings
warnings.filterwarnings('ignore')
import evaluate
import os
import pickle
from tqdm import tqdm
import torch
import config
import json
import argparse
import numpy as np
from collections import Counter

from sentence_transformers import SentenceTransformer
from utils import (
    MODEL_PATH_DICT,
    set_seed,
    compute_self_certainty_scores, 
    get_self_certainty_sample, 
    compute_weighted_mean, 
    clean_generation, 
    clean_answer, 
    is_correct
    )


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


def main(args):
    experiment_id = os.getpid()
    cache_dir = f"/tmp/rouge_cache_{experiment_id}"
    os.environ['HF_EVALUATE_CACHE'] = cache_dir
    rouge = evaluate.load('rouge', experiment_id=experiment_id, cache_dir=cache_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer(args.embed_model).to(device)

    # --- Load generations ---
    gen_path = f"{config.output_dir}/{args.dataset}__{args.model}__{args.n_samples}__generation.pkl"
    with open(gen_path, "rb") as infile:
        generations = pickle.load(infile)

    accuracy = {}
    
    for i, gen in enumerate(tqdm(generations, desc="Processing generations")):
        
        # --- Find the least uncertain samples ---
        cleaned_texts = gen["cleaned_generated_texts"]
        samples_avg_nll = gen["samples_avg_nll"]
        samples_nll = gen["samples_nll"]
        
        # --- RDS score ---
        embeddings = embed_model.encode(cleaned_texts, convert_to_tensor=True, device=device)
        mean_embedding = torch.mean(embeddings, dim=0)
        diffs = embeddings - mean_embedding 
        rds_scores = torch.norm(diffs, p=1, dim=1)
        
        probs = np.exp(-np.array(samples_avg_nll))
        probs /= probs.sum() 
        weighted_mean_embeddings = compute_weighted_mean(
            embeddings, torch.tensor(probs, dtype=torch.float32, device=device)
        )
        
        diffs_weighted = embeddings - weighted_mean_embeddings.unsqueeze(0)
        weighted_rds_scores = torch.norm(diffs_weighted, p=1, dim=1)
        
        # --- Ranking and find samples ---
        nll_sample = cleaned_texts[np.argmin(samples_nll)]
        avg_nll_sample = cleaned_texts[np.argmin(samples_avg_nll)]
        rds_sample = cleaned_texts[torch.argmin(rds_scores).item()]
        weighted_rds_sample = cleaned_texts[torch.argmin(weighted_rds_scores).item()]

        freq = Counter(cleaned_texts)
        majority_sample = freq.most_common(1)[0][0]
        
        if args.self_certainty:
            sc_cache_path = f"{config.output_dir}/{args.dataset}__{args.model}__self_certainty.pkl"
            os.makedirs(os.path.dirname(sc_cache_path), exist_ok=True)

            if os.path.exists(sc_cache_path):
                with open(sc_cache_path, "rb") as f:
                    all_self_certainty = pickle.load(f)
            else:
                prompts = [gen["prompt"] for gen in generations]
                generated_texts_list = [gen["cleaned_generated_texts"] for gen in generations]
                
                all_self_certainty = compute_self_certainty_scores(
                    model_dir=MODEL_PATH_DICT[args.model],
                    prompts=prompts,
                    generated_texts_list=generated_texts_list,
                    batch_size=4,
                    device=device
                )
                
                with open(sc_cache_path, "wb") as f:
                    pickle.dump(all_self_certainty, f)
                print(f"Saved self-certainty scores to {sc_cache_path}")
                
            gen["samples_ce"] = all_self_certainty[i]
        
        # --- Self-certainty sample ---
        if "samples_ce" in gen:
            sc_scores = np.array(gen["samples_ce"])
            self_certainty_sample = get_self_certainty_sample(sc_scores, cleaned_texts)
        else:
            self_certainty_sample = None

        # --- Evaluation ---
        methods = ['nll', 'avg_nll', 'rds', 'weighted_rds', 'majority'] if self_certainty_sample is None else \
                  ['nll', 'avg_nll', 'rds', 'weighted_rds', 'majority', 'self_certainty']
        samples = [nll_sample, avg_nll_sample, rds_sample, weighted_rds_sample, majority_sample] if self_certainty_sample is None else \
                  [nll_sample, avg_nll_sample, rds_sample, weighted_rds_sample, majority_sample, self_certainty_sample]
        for method, sample in zip(methods, samples):
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
    json_path = f"{config.result_dir}/{args.dataset}__{args.model}__ranking.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"📄 Saved results summary to {json_path}")

    return results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Best-of-N accuracy for different ranking methods')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--embed_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--self_certainty', type=bool, default=False)
    args = parser.parse_args()

    set_seed(10)
    main(args)