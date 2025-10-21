from vllm import LLM, SamplingParams
import os
import argparse
import pickle
import random
import numpy as np
import torch
import tqdm
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import config
import pandas as pd
from datasets import load_dataset
import json

# --------------------
# Utility setup
# --------------------
def set_seed(seed_value=10):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


MODEL_PATH_DICT = {
    "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "falcon3-1b": "tiiuae/falcon3-1b-instruct",
    "falcon3-7b": "tiiuae/falcon3-7b-instruct",
    "falcon3-10b": "tiiuae/falcon3-10b-instruct",
    "gemma-7b": "google/gemma-7b-it",
    "gemma-2b": "google/gemma-2b-it",
    "gemma2-2b": "google/gemma-2-2b-it",
    "gemma2-27b": "google/gemma-2-27b",
    "gemma2-9b": "google/gemma-2-9b-it",
    "gemma3-1b": "google/gemma-3-1b-it",
    "gemma3-4b": "google/gemma-3-4b-it",
    "phi3-7b": "microsoft/Phi-3-small-8k-instruct",
    "phi3-3b": "microsoft/Phi-3-mini-4k-instruct",
    "phi3.5-3b": "microsoft/Phi-3.5-mini-instruct",
    "phi4-3b": "microsoft/Phi-4-mini-instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral-small": "mistralai/Mistral-Small-Instruct-2409",
    "mistral-large": "mistralai/Mistral-Large-Instruct-2407"
}

   
# --------------------
# PARSE DATASET
# --------------------
def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def parse_dataset(args):
    
    if args.dataset in ['sciq', 'nq']:
        question_file = f"{config.data_dir}/{args.dataset}.txt"
        
        if not os.path.exists(question_file):
            raise FileNotFoundError(f"Question file not found: {question_file}")
        
        questions, answers = [], []
        with open(question_file, "r", encoding="utf-8") as f:
            blocks = f.read().strip().split("\n\n")

        for item in blocks:
            lines = item.strip().split("\n")
            if len(lines) < 2:
                continue
            question = lines[0].strip()
            ans = [a.strip() for a in lines[1].split(";") if a.strip()]
            questions.append(question)
            answers.append(ans)

    elif args.dataset == 'svamp':

        ds = load_dataset("ChilleD/SVAMP")['test']
        questions = [item['question_concat'] for item in ds]
        answers = [item['Answer'] for item in ds]
    
    elif args.dataset == 'trivia_qa':
        val_data = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        train_data = load_dataset("trivia_qa", "rc.nocontext", split="train")
        data_for_few_shot_prompt = train_data.select(range(0, 10))

        few_shot_prompt = 'This is a bot that correctly answers questions. \n'
        for sample in data_for_few_shot_prompt:
            few_shot_prompt += 'Question: ' + sample['question'] + ' Answer: ' + sample['answer']['value'] + ' '
            
        questions = [item['question'] for item in val_data]
        answers = [item['answer']['value'] for item in val_data] 
        
    elif args.dataset == 'coqa':
        with open(f'{config.data_dir}/coqa.json', 'r') as infile:
            data = json.load(infile)['data']

        questions, answers, stories = [], [], []

        for sample in data:
            story = sample['story']
            list_questions = sample['questions']
            list_answers = sample['answers']
            for question_index, question in enumerate(list_questions):
                
                question = question['input_text']
                answer = list_answers[question_index]['input_text']
                story = story + '\nQuestion: ' + question + '\nAnswer: ' + answer

                questions.append(story + f'\nQuestion: {question}\nAnswer: ')
                answers.append(answer)
                
            stories.append(story)

    elif args.dataset == 'gsm8k':
        ds = load_dataset("gsm8k", "main")['test']
        questions = [item['question'] for item in ds]
        answers = [item['answer'] for item in ds]
    else:
        raise ValueError(f"Dataset {args.dataset} not supported for parsing.")

    # Build few-shot prompt
    if args.dataset == 'gsm8k':
        few_shot_prompt = "Please reason step by step, and put your final answer after ####.\n\n"
    else:
        few_shot_prompt = "This is a bot that correctly answers questions.\n\n"
    n_few = min(args.few_shot_num, len(questions))
    for i in range(n_few):
        if args.dataset in ['sciq', 'nq']:
            few_shot_prompt += f"Question: {questions[i]}\nAnswer: {answers[i][0]}\n\n"
        else:
            few_shot_prompt += f"Question: {questions[i]}\nAnswer: {answers[i]}\n\n"
    
    if args.dataset == 'coqa':
        few_shot_prompt = f"This is a bot that correctly answers questions based on the provided context.\n\n{stories[0]}\n\n"

    # Construct processed dataset
    processed_dataset = []
    if args.dataset == 'coqa':
        for i in range(n_few, len(questions)):
            prompt = few_shot_prompt + questions[i]
            processed_dataset.append({
                "question": questions[i],
                "answer": answers[i],
                "prompt": prompt
            })
    else:
        for i in range(n_few, len(questions)):
            prompt = few_shot_prompt + f"Question: {questions[i]}\nAnswer:"
            processed_dataset.append({
                "question": questions[i],
                "answer": answers[i],
                "prompt": prompt
            })

    return processed_dataset

# --------------------
# GENERATION + NLL
# --------------------
def flatten_logprobs(logprobs):
    """
    Flattens nested list[dict[token_id -> Logprob]] into a list of floats.
    """
    flat = []
    if not logprobs:
        return flat
    if isinstance(logprobs, list):
        for step in logprobs:
            if isinstance(step, dict):
                flat.extend([v.logprob for v in step.values()])
    elif isinstance(logprobs, dict):
        flat.extend([v.logprob for v in logprobs.values()])
    return flat
    
def generate_sequences(llm, dataset, rouge, args):
    
    print('--- GENERATION PARAMETERS ---')
    print('Dataset:', args.dataset)
    print('Model:', args.model)
    
    greedy_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0,
        n=1,
        logprobs=1
    )
    multinomial_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=1,
        n=args.n_samples,
        logprobs=1
    )

    sequences = []
    for i, batch in enumerate(tqdm.tqdm(dataset)):
        prompt = batch['prompt']
        question = batch['question']
        answer = batch['answer']

        # === GREEDY DECODING ===
        greedy_out = llm.generate(prompt, sampling_params=greedy_params)[0].outputs[0]
        if args.dataset == 'gsm8k':
            greedy_text = greedy_out.text.strip()
        else:
            greedy_text = clean_generation(greedy_out.text)
            
        greedy_logprobs = greedy_out.logprobs
        
        if args.dataset in ['gsm8k', 'svamp']: # exact match for math datasets
            if args.dataset == 'svamp':
                eval_score = greedy_text.strip() == answer.strip()
            else:
                eval_score = extract_hash_answer(greedy_text) == extract_hash_answer(answer)
        else:
            eval_score = rouge.compute(predictions=[greedy_text], references=[answer])['rougeL']

        # === MULTINOMIAL DECODING ===
        sampled_outputs = llm.generate(prompt, sampling_params=multinomial_params)[0].outputs
        generated_texts = [o.text for o in sampled_outputs]
        generation_logprobs = [o.logprobs for o in sampled_outputs]

        # === CLEANING ===
        cleaned = [clean_generation(g) for g in generated_texts]

        # === UNCERTAINTY (negative log-likelihood) ===
        samples_avg_nll, samples_nll = [], []
        
        for sample in generation_logprobs:
            flat = flatten_logprobs(sample)
            avg_nll = -np.mean(flat) if len(flat) > 0 else np.nan
            nll = -np.sum(flat) if len(flat) > 0 else np.nan
            samples_avg_nll.append(avg_nll)
            samples_nll.append(nll)

        greedy_flat = flatten_logprobs(greedy_logprobs)
        greedy_avg_nll = -np.mean(greedy_flat) if greedy_flat else np.nan
        greedy_nll = -np.sum(greedy_flat) if greedy_flat else np.nan

        # === DEBUG PRINTS ===
        if i < 5:
            print('Prompt:', prompt)
            print('Question:', question)
            print('Answer:', answer)
            print('Greedy text:', greedy_text)
            print('Greedy logprobs:', greedy_logprobs)
            print('Samples avg NLL:', samples_avg_nll)
            print('Samples NLL:', samples_nll)
            print('Eval score:', eval_score)
            print('---')

        # === STRUCTURE OUTPUT ===
        sequences.append({
            'id': f"{args.dataset}_{i}",
            'prompt': prompt,
            'question': question,
            'answer': answer,
            'generated_texts': generated_texts,
            'cleaned_generated_texts': cleaned,
            'samples_nll': samples_nll,
            'samples_avg_nll': samples_avg_nll,
            'greedy_text': greedy_text,
            'greedy_nll': greedy_nll,
            'greedy_avg_nll': greedy_avg_nll,
            'eval_score': eval_score
        })
        
    return sequences


def clean_generation(text):
    for s in ['.', '\n', 'Q:', 'A:', 'question:', 'answer:', 'Question:', 'Answer:', 'Questions:', 'questions:', 'QUESTION:', 'ANSWER:', ':']:
        if s in text:
            text = text.split(s)[0].rstrip()
    return text


# --------------------
# SEMANTIC SIMILARITY
# --------------------
def compute_semantic_similarity(sequences, semantic_model, semantic_tokenizer, device):
    
    for sample in tqdm.tqdm(sequences):
        question = sample['question']
        generations = sample['cleaned_generated_texts']
        unique_generations = list(set(generations))
        semantic_set_ids = {ans: i for i, ans in enumerate(unique_generations)}

        # pairwise DeBERTa similarity
        for i, a1 in enumerate(unique_generations):
            for j, a2 in enumerate(unique_generations):
                if j <= i:
                    continue
                qa1 = question + " " + a1
                qa2 = question + " " + a2

                # NLI prediction: 0 = contradiction, 1 = neutral, 2 = entailment
                encoded = semantic_tokenizer(qa1, qa2, return_tensors='pt', truncation=True, max_length=512).to(device)
                logits = semantic_model(**encoded).logits
                pred = torch.argmax(logits, dim=1).item()

                encoded_rev = semantic_tokenizer(qa2, qa1, return_tensors='pt', truncation=True, max_length=512).to(device)
                logits_rev = semantic_model(**encoded_rev).logits
                pred_rev = torch.argmax(logits_rev, dim=1).item()

                if not(pred == 0 or pred_rev == 0):
                    semantic_set_ids[a2] = semantic_set_ids[a1]

        list_of_semantic_set_ids = [semantic_set_ids[x] for x in generations]
        sample['semantic_set_ids'] = list_of_semantic_set_ids

    return sequences


# --------------------
# MAIN EVALUATION PIPELINE
# --------------------
def main(args):
    experiment_id = os.getpid()
    cache_dir = f"/tmp/rouge_cache_{experiment_id}"
    os.environ['HF_EVALUATE_CACHE'] = cache_dir
    rouge = evaluate.load('rouge', experiment_id=experiment_id, cache_dir=cache_dir)
    
    # Load dataset
    dataset = parse_dataset(args=args)

    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset[: int(len(dataset) * args.fraction_of_data_to_use)]

    # Init model
    hf_model_dir = MODEL_PATH_DICT[args.model]
    llm = LLM(model=hf_model_dir, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=2048)

    # Run generation
    sequences = generate_sequences(llm=llm, dataset=dataset, rouge=rouge, args=args)
    
    if args.other_baselines:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        semantic_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        semantic_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)
        sequences = compute_semantic_similarity(sequences=sequences, semantic_model=semantic_model, semantic_tokenizer=semantic_tokenizer, device=device)

    # Save
    output_path = f"{config.output_dir}/{args.dataset}__{args.model}__generation.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(sequences, f)

    print(f"Saved results to {output_path}")
    return sequences

# --------------------
# CLI ENTRY
# --------------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
    parser.add_argument('--few_shot_num', type=int, default=5)
    parser.add_argument('--model', type=str, default='gemma-7b', required=True)
    parser.add_argument('--dataset', type=str, default='coqa', required=True)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--other_baselines', type=bool, default=False)
    
    args = parser.parse_args()
    set_seed(10)
    main(args)