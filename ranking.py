
from datetime import datetime
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
import pandas as pd

import logging
logging.basicConfig(level=logging.ERROR)

from sentence_transformers import SentenceTransformer
from utils import (
    MODEL_PATH_DICT,
    set_seed,
    compute_self_certainty_scores, 
    get_self_certainty_sample, 
    compute_weighted_mean, 
    clean_generation, 
    clean_answer, 
    is_correct,
    compute_label,
    extract_math_response
    )


def evaluation_sample(dataset, text, answer, rouge, question=None, eval_method="rougeL", api_type="cohere", threshold=0.3):
    
    # if dataset in ['gsm8k']:
    #     text = extract_math_response(text=text, args=args)
    # else:
    #     text = clean_generation(text)
        
    if dataset in ['svamp', 'arith']: # exact match for math datasets
        eval_score = compute_label(generation=text, ground_truth=answer, eval_method="exact_match")
    elif dataset in ['gsm8k']:
        eval_score = int(text == np.round(answer, 1))
        # model_answer = clean_answer(text)
        # eval_score = is_correct(model_answer=model_answer, answer=answer)
    elif dataset in ['formal_logic']:
        eval_score = int(text == answer)
    else:
        eval_score = compute_label(generation=text, ground_truth=answer, question=question, eval_method=eval_method, rouge=rouge, api_type=api_type)
    
    if dataset in ['gsm8k', 'svamp', 'arith']:
        acc = int(eval_score == 1.0)
    else:
        if eval_method == "rougeL":
            acc = int(eval_score > threshold)
        else:
            acc = int(eval_score)
        
    return acc


def main(args):
    experiment_id = os.getpid()
    cache_dir = f"/tmp/rouge_cache_{experiment_id}"
    os.environ['HF_EVALUATE_CACHE'] = cache_dir
    rouge = evaluate.load('rouge', experiment_id=experiment_id, cache_dir=cache_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer(args.embed_model).to(device)

    # --- Load generations ---
    gen_path = f"{config.output_dir}/{args.dataset}_{args.model}_N={args.n_samples}_F={args.fraction_of_data_to_use}_A={args.api_type}_S={args.seed}__generation.pkl"
    with open(gen_path, "rb") as infile:
        generations = pickle.load(infile)

    accuracy = {}
    
    for i, gen in enumerate(tqdm(generations, desc="Processing generations")):
        
        # --- Find the least uncertain samples ---
        cleaned_texts = gen["cleaned_generated_texts"]
        extracted_answers = gen["extracted_answers"] if "extracted_answers" in gen else [None] * len(cleaned_texts)
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
        if args.dataset in ['gsm8k', 'formal_logic']:
        
            nll_sample = extracted_answers[np.argmin(samples_nll)]
            avg_nll_sample = extracted_answers[np.argmin(samples_avg_nll)]
            rds_sample = extracted_answers[torch.argmin(rds_scores).item()]
            weighted_rds_sample = extracted_answers[torch.argmin(weighted_rds_scores).item()]
            
            freq = Counter(extracted_answers)
            majority_sample = freq.most_common(1)[0][0]
        else:
            nll_sample = cleaned_texts[np.argmin(samples_nll)]
            avg_nll_sample = cleaned_texts[np.argmin(samples_avg_nll)]
            rds_sample = cleaned_texts[torch.argmin(rds_scores).item()]
            weighted_rds_sample = cleaned_texts[torch.argmin(weighted_rds_scores).item()]

            freq = Counter(cleaned_texts)
            majority_sample = freq.most_common(1)[0][0]
        
        if args.self_certainty:
            sc_cache_path = f"{config.output_dir}/{args.dataset}_{args.model}_N={args.n_samples}_F={args.fraction_of_data_to_use}_A={args.api_type}_S={args.seed}__self_certainty.pkl"
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
        elif args.deep_conf:
            # TODO: Add DeepConf code here
            pass
        
        # --- Self-certainty sample ---
        if "samples_ce" in gen:
            sc_scores = np.array(gen["samples_ce"])
            if args.dataset in ['gsm8k', 'formal_logic']:
                self_certainty_sample = get_self_certainty_sample(sc_scores, extracted_answers)
            else:
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
                question=gen["question"] if "question" in gen else None,
                rouge=rouge,
                api_type=args.api_type,
                eval_method=args.eval_method,
                threshold=args.threshold
            )
            
            if method not in accuracy:
                accuracy[method] = []
                
            accuracy[method].append(acc)
            
            # For debug
            if i < 3:
                print(f"Sample {i} | Method: {method} | Acc: {acc} | Sample: {sample}...")
    
    # --- Reporting ---
    results = {}
    print("\n=== Metric Performance ===")
    for method, values in accuracy.items():
        final_acc = np.mean(values)
        results[method] = round(final_acc, 4)
        print(f"{method:55s} → ACC: {final_acc:.4f}")

    # --- Prepare row to append ---
    row = {
        "timestamp": args.timestamp,
        "dataset": args.dataset,
        "model": args.model,
        "embed_model": args.embed_model,
        "n_samples": args.n_samples,
        "threshold": args.threshold,
        "eval_method": args.eval_method,
        "api_type": args.api_type,
        "fraction_of_data_to_use": args.fraction_of_data_to_use,
        "seed": args.seed,
    }
    row.update(results)
    new_row_df = pd.DataFrame([row])

    # --- Check if file exists ---
    tsv_file = f'results/ranking_logs.tsv'
    if os.path.exists(tsv_file):
        df = pd.read_csv(tsv_file, sep='\t')
        
        # Union all columns
        all_cols = sorted(set(df.columns).union(new_row_df.columns))

        df = df.reindex(columns=all_cols)
        new_row_df = new_row_df.reindex(columns=all_cols)

        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        df = new_row_df

    # --- Save back ---
    df.to_csv(tsv_file, sep='\t', index=False)

    return results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Best-of-N accuracy for different ranking methods')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--embed_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--self_certainty', action='store_true', help='Whether to compute self-certainty scores')
    parser.add_argument('--deep_conf', action='store_true', help='Whether to compute DeepConf scores')
    parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0, help='Fraction of data to use for evaluation (for quick testing)')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for binary classification of correctness (used for non-math datasets)')
    parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
    parser.add_argument('--eval_method', type=str, default='rougeL', help='Evaluation method for non-math datasets (e.g., rougeL or api)')
    parser.add_argument('--api_type', type=str, default='cohere', choices=['gemini', 'cohere'], help='API type for LLM evaluation')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    args.timestamp = timestamp

    print(f"RANKING: Dataset={args.dataset}, Model={args.model}, N={args.n_samples}, F={args.fraction_of_data_to_use}, T={args.threshold}, S={args.seed}, E={args.eval_method}, A={args.api_type}.")
    set_seed(args.seed)
    start_time = datetime.now()
    main(args)
    end_time = datetime.now()
    print(f"Total evaluation time: {end_time - start_time}")