import numpy as np
import torch

# -------------------------------------
### Semantic Entropy
# -------------------------------------
# Compute semantic similarity set IDs
def compute_semantic_similarity(sample, semantic_model, semantic_tokenizer, device='cuda:0'):

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

    return list_of_semantic_set_ids

# Main Semantic Entropy Computation
def compute_semantic_entropy(sample, embed_model, embed_tokenizer, device='cuda:0'):
    """Compute semantic entropy for a single sample.
    Args:
        sample (dict): Dictionary containing 'samples_avg_nll' and 'cleaned_generated_texts'.
        embed_model: Transformer model for embedding and prediction.
        embed_tokenizer: Tokenizer for encoding inputs.
    """
    # Semantic set
    semantic_set_ids = compute_semantic_similarity(sample, embed_model, embed_tokenizer, device)
    
    # Convert inputs to tensors
    avg_nll = torch.as_tensor(sample['samples_avg_nll'], dtype=torch.float32)
    semantic_set_ids = torch.as_tensor(semantic_set_ids, dtype=torch.int64)

    # Get unique semantic set IDs (excluding -1)
    valid_set_ids = torch.unique(semantic_set_ids[semantic_set_ids != -1])
    
    if len(valid_set_ids) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=avg_nll.device)
    
    # Aggregate log-likelihoods for each semantic set
    aggregated_log_probs = []
    for set_id in valid_set_ids:
        mask = (semantic_set_ids == set_id)
        agg_log_prob = torch.logsumexp(avg_nll[mask], dim=0)
        aggregated_log_probs.append(agg_log_prob)
    
    # Convert to tensor and compute probabilities
    aggregated_log_probs = torch.tensor(aggregated_log_probs, dtype=torch.float32, device=avg_nll.device)
    probs = torch.softmax(aggregated_log_probs, dim=0)  # Normalize to probabilities
    
    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=0)  # Add epsilon to avoid log(0)
    
    return entropy.item()

# -------------------------------------
### Compute DEg and Semantic Density
# -------------------------------------
def compute_deg_semantic_density(sample, embed_model, embed_tokenizer, device='cuda:0'):
    """Compute contradiction probability (deg) and semantic density for a single sample as scalars.
    
    Args:
        sample (dict): Dictionary containing 'question', 'cleaned_generated_texts', 
                       'greedy_text', and 'samples_avg_nll'.
        embed_model: Transformer model for embedding and prediction.
        embed_tokenizer: Tokenizer for encoding inputs.
        device (str): Device to run computations on (default: 'cuda:0').
    
    Returns:
        tuple: (deg, sd), where deg and sd are Python scalars (float).
    """
    question = sample['question']
    cleaned_generated_texts = sample['cleaned_generated_texts']
    most_likely_text = sample['greedy_text']
    contradict_prob_list = []

    likelihood_sum = 0.0
    semantic_density = 0.0
    
    # Evaluate semantic similarity
    unique_cleaned_generation = set()
    unique_index = []

    for generation_index in range(len(cleaned_generated_texts)):
        generation_text = cleaned_generated_texts[generation_index]
        if generation_text not in unique_cleaned_generation:
            unique_cleaned_generation.add(generation_text)
            unique_index.append(generation_index)

    # Semantic Density & Deg matrix
    for generation_index in unique_index:
        qa_1 = question + ' ' + cleaned_generated_texts[generation_index]
        qa_2 = question + ' ' + most_likely_text
        average_likelihood = float(np.exp(-sample['samples_avg_nll'][generation_index]))
        origin_input = qa_1 + ' [SEP] ' + qa_2

        # Encode and predict for forward input
        encoded_input = embed_tokenizer(origin_input, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            prediction = embed_model(**encoded_input).logits[0]
        
        # Apply torch.softmax and convert to scalars
        prediction_softmax = torch.softmax(prediction, dim=-1)
        contradict_prob_1 = float(1 - prediction_softmax[2].item())
        semantic_distance = float(prediction_softmax[0].item() + 0.5 * prediction_softmax[1].item())
        semantic_density += 0.5 * (1.0 - semantic_distance) * average_likelihood

        # Encode and predict for reverse input
        reverse_input = qa_2 + ' [SEP] ' + qa_1
        encoded_reverse_input = embed_tokenizer(reverse_input, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            reverse_prediction = embed_model(**encoded_reverse_input).logits[0]
        
        # Apply torch.softmax and convert to scalars
        reverse_prediction_softmax = torch.softmax(reverse_prediction, dim=-1)
        contradict_prob_2 = float(1 - reverse_prediction_softmax[2].item())
        reverse_semantic_distance = float(reverse_prediction_softmax[0].item() + 0.5 * reverse_prediction_softmax[1].item())
        
        # Update metrics
        semantic_density += 0.5 * (1.0 - reverse_semantic_distance) * average_likelihood
        likelihood_sum += average_likelihood
        contradict_prob_list.append((contradict_prob_1 + contradict_prob_2) / 2.0)
    
    # Compute final metrics as scalars
    deg = np.mean(contradict_prob_list) if contradict_prob_list else 0.0
    sd = 1 - semantic_density / likelihood_sum if likelihood_sum > 0 else 1

    return deg, sd