import numpy as np
import torch
import re

ANSWER_TRIGGER = 'The answer is'
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

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

def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append("There are 15 trees in the grove. "
                    "Grove workers will plant trees in the grove today. "
                    "After they are done, there will be 21 trees. "
                    "How many trees did the grove workers plant today?")
    chain.append("There are 15 trees originally. "
                 "Then there were 21 trees after some more were planted. "
                 "So there must have been 21 - 15 = 6.")
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?")
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?")
    chain.append("Originally, Leah had 32 chocolates. "
                 "Her sister had 42. So in total they had 32 + 42 = 74. "
                 "After eating 35, they had 74 - 35 = 39.")
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?")
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.")
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?")
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.")
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?")
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29.")
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?")
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls.")
    answer.append("33")

    question.append("Olivia has $23. She bought five bagels for $3 each. "
                    "How much money does she have left?")
    chain.append("Olivia had 23 dollars. "
                 "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                 "So she has 23 - 15 dollars left. 23 - 15 is 8.")
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Question: " + question[i] + "\nAnswer: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text


def compute_wasserstein(norms, weights, p=2):
    """Compute generalized p-Wasserstein distance."""
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(norms.device if isinstance(norms, torch.Tensor) else 'cpu')
    return ((weights_tensor * (norms ** p)).sum()) ** (1.0 / p)


def compute_eigen_embed(sentence_embeddings, alpha=1e-3):
    embeddings = (
        sentence_embeddings.cpu().numpy() 
        if isinstance(sentence_embeddings, torch.Tensor) 
        else np.array(sentence_embeddings)
    )
    if embeddings.ndim != 2:
        raise ValueError(f"Expected (N, D) tensor, got {embeddings.shape}")
    
    embeddings = np.array(embeddings)  # Shape: (N, D)
    N = embeddings.shape[0]
    
    # CRUCIAL: Use sample covariance (N x N), NOT feature covariance
    cov = np.cov(embeddings) # Shape: (N, N)
    cov = cov + alpha * np.eye(N)  # Regularize

    # SVD
    u, s, vT = np.linalg.svd(cov)  # singular values
    s = np.sort(s)[::-1]  # descending order
    eigen_score = np.mean(np.log10(s))

    return eigen_score

def eigen_score(prompt, generations, tokenizer=None, model=None, sentence_embeddings=None, mode="internal", variance_dim='feature'):
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


def eigen_score_refactor(prompt, tokenizer=None, model=None, sentence_embeddings=None, mode="internal", 
                         n_samples=10, max_new_tokens=32, top_p=0.99, top_k=10, temperature=1.0, variance_dim='feature'):
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
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # num_tokens = inputs["input_ids"].shape[1]
        # token_idx = max(num_tokens - 2, 0) # based on author's implementation

        # Generate with hidden states
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=1,
                num_return_sequences=n_samples,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            
        # hidden_states: list[num_steps][num_layers][batch, seq_len, hidden_dim]
        hidden_states_all = outputs.hidden_states
        layer_idx = len(hidden_states_all[0]) // 2
        
        # step_states[layer_idx]: shape [batch, 1, hidden_dim]
        for hidden_states in hidden_states_all:
            emb = hidden_states[layer_idx][:, -1, :].squeeze().cpu().numpy()
            
            if emb.ndim == 1:
                emb = emb[None, :]  # shape [1, hidden]  
            embeddings.append(emb)
        
        embeddings = np.concatenate(embeddings, axis=0)  # shape [n_samples, hidden_dim]
    
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
    
    return (-1) * eigen_score


# ----------------
### EigenScore (exactly like original author implementation)
# ----------------
def getEigenIndicator_v0(hidden_states, num_tokens): 
    alpha = 1e-3
    selected_layer = int(len(hidden_states[0])/2)
    # selected_layer = -1
    if len(hidden_states) < 2:
        return 0
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for ind in range(hidden_states[1][-1].shape[0]):
        last_embeddings[ind,:] = hidden_states[num_tokens[ind]-2][selected_layer][ind,0,:]
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator


def get_num_tokens(generation):  # generation: num_seq x max(num_tokens)
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        num_tokens.append(count+1)
    return num_tokens

def eigen_score_origin_v2(
    prompt=None,
    tokenizer=None,
    model=None,
    sentence_embeddings=None,
    mode="internal",
    n_samples=10,
    max_new_tokens=64,
    top_p=0.99,
    top_k=10,
    temperature=1.0
):
    """
    Compute EigenScore exactly like the original author implementation.
    
    - mode="internal": Use LLM hidden states (at token position = total_tokens - 2)
    - mode="third_party": Use precomputed sentence embeddings (N x D)
    
    Returns: -mean(log10(singular_values))  (higher = more diverse = more uncertain)
    """
    alpha = 1e-3
    embeddings = []

    # ================================
    # 1. INTERNAL MODE: Generate + extract hidden states
    # ================================
    if mode == "internal":
        if tokenizer is None or model is None:
            raise ValueError("tokenizer and model are required for mode='internal'")
        if prompt is None:
            raise ValueError("prompt is required for mode='internal'")

        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=1,
                num_return_sequences=n_samples,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # outputs.sequences: [n_samples, total_len]
        # outputs.hidden_states: list[step] of tuple[layer](batch, seq, dim)
        sequences = outputs.sequences
        hidden_states_all = outputs.hidden_states  # len = max_new_tokens
        input_length = inputs['input_ids'].shape[1]
        generation = outputs.sequences[:, input_length:].cpu()
        num_tokens = get_num_tokens(generation)

        eigen_score = getEigenIndicator_v0(hidden_states_all, num_tokens)
        
        return eigen_score

    # ================================
    # 2. THIRD-PARTY EMBEDDING MODE
    # ================================
    elif mode == "third_party":
        if sentence_embeddings is None:
            raise ValueError("sentence_embeddings required for mode='third_party'")
        embeddings = (
            sentence_embeddings.cpu().numpy() 
            if isinstance(sentence_embeddings, torch.Tensor) 
            else np.array(sentence_embeddings)
        )
        if embeddings.ndim != 2:
            raise ValueError(f"Expected (N, D) tensor, got {embeddings.shape}")
        
        embeddings = np.array(embeddings)  # Shape: (N, D)
        N = embeddings.shape[0]
        
        if N < 2:
            return 0.0  # or raise warning

        # CRUCIAL: Use sample covariance (N x N), NOT feature covariance
        cov = np.cov(embeddings) # Shape: (N, N)
        cov = cov + alpha * np.eye(N)  # Regularize

        # SVD
        u, s, vT = np.linalg.svd(cov)  # singular values
        eigen_score = np.mean(np.log10(s))
        
        return eigen_score  # Negative → higher diversity = higher uncertainty

    else:
        raise ValueError("mode must be 'internal' or 'third_party'")


def eigen_score_origin(
    prompt=None,
    tokenizer=None,
    model=None,
    sentence_embeddings=None,
    mode="internal",
    n_samples=10,
    max_new_tokens=64,
    top_p=0.99,
    top_k=10,
    temperature=1.0
):
    """
    Compute EigenScore exactly like the original author implementation.
    
    - mode="internal": Use LLM hidden states (at token position = total_tokens - 2)
    - mode="third_party": Use precomputed sentence embeddings (N x D)
    
    Returns: -mean(log10(singular_values))  (higher = more diverse = more uncertain)
    """
    alpha = 1e-3
    embeddings = []

    # ================================
    # 1. INTERNAL MODE: Generate + extract hidden states
    # ================================
    if mode == "internal":
        if tokenizer is None or model is None:
            raise ValueError("tokenizer and model are required for mode='internal'")
        if prompt is None:
            raise ValueError("prompt is required for mode='internal'")

        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=1,
                num_return_sequences=n_samples,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # outputs.sequences: [n_samples, total_len]
        # outputs.hidden_states: list[step] of tuple[layer](batch, seq, dim)
        sequences = outputs.sequences
        hidden_states_all = outputs.hidden_states  # len = max_new_tokens

        # Middle layer
        layer_idx = len(hidden_states_all[0]) // 2  # hidden_states_all[0] = tuple of layers

        for i in range(n_samples):
            seq = sequences[i]
            seq_len = seq.shape[0]
            prompt_len = inputs['input_ids'].shape[1]

            # Nếu generation rỗng hoặc quá ngắn
            if seq_len <= prompt_len:
                # Dùng hidden state cuối cùng của prompt
                emb = hidden_states_all[0][layer_idx][i, -1, :].cpu().numpy()
                embeddings.append(emb)
                continue

            # Token trước EOS
            token_idx = seq_len - 2
            if token_idx < 0:
                token_idx = 0

            # Bước generate
            step_idx = token_idx - prompt_len
            step_idx = max(0, min(step_idx, len(hidden_states_all) - 1))

            # Lấy hidden state
            hidden = hidden_states_all[step_idx][layer_idx]  # [B, seq_len_step, D]
            actual_seq_len = hidden.shape[1]

            # Clamp token_idx
            token_idx_in_step = token_idx - (prompt_len if step_idx == 0 else 0)
            token_idx_in_step = max(0, min(token_idx_in_step, actual_seq_len - 1))

            emb = hidden[i, token_idx_in_step, :].cpu().numpy()
            embeddings.append(emb)

    # ================================
    # 2. THIRD-PARTY EMBEDDING MODE
    # ================================
    elif mode == "third_party":
        if sentence_embeddings is None:
            raise ValueError("sentence_embeddings required for mode='third_party'")
        embeddings = (
            sentence_embeddings.cpu().numpy() 
            if isinstance(sentence_embeddings, torch.Tensor) 
            else np.array(sentence_embeddings)
        )
        if embeddings.ndim != 2:
            raise ValueError(f"Expected (N, D) tensor, got {embeddings.shape}")

    else:
        raise ValueError("mode must be 'internal' or 'third_party'")

    # ================================
    # 3. FINAL: Compute EigenScore (N x N covariance)
    # ================================
    embeddings = np.array(embeddings)  # Shape: (N, D)
    N = embeddings.shape[0]
    
    if N < 2:
        return 0.0  # or raise warning

    # CRUCIAL: Use sample covariance (N x N), NOT feature covariance
    cov = np.cov(embeddings) # Shape: (N, N)
    cov = cov + alpha * np.eye(N)  # Regularize

    # SVD
    u, s, vT = np.linalg.svd(cov)  # singular values
    eigen_score = np.mean(np.log10(s))
    

    return -eigen_score  # Negative → higher diversity = higher uncertainty
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
    # nll_probs /= nll_probs.sum()
    top_probs = np.sort(nll_probs)[::-1]
    filtered = top_probs[top_probs >= alpha]
    if len(filtered) == 0:
        filtered = np.array([top_probs[0]])
        
    return approx(filtered)

# -------------------------------------
### New clustering methods
# -------------------------------------
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import torch
from collections import defaultdict

# -------------------------------
# CLUSTERING FUNCTIONS
# -------------------------------
def compute_clusters(sample, embed_model=None, device='cuda:0',
                     cluster_mode='pca_then_kmeans',
                     max_components=10, max_k=10,
                     semantic_model=None, semantic_tokenizer=None):
    """
    Compute cluster assignments using different methods:
      - 'pca_only'         → PCA projections (1..max_components)
      - 'kmeans_only'      → KMeans (k=1..max_k)
      - 'pca_then_kmeans'  → PCA (n_components=1..max_components) then KMeans (k=1..max_k)
      - 'semantic_clustering' → Clustering by semantic equivalence using NLI

    Returns:
      dict {config_key: cluster_ids}
        e.g., 'pca3', 'kmeans5', 'pca3_k5', 'semantic'
    """
    generations = sample['cleaned_generated_texts']
    all_cluster_assignments = {}

    if cluster_mode in ['pca_only', 'kmeans_only', 'pca_then_kmeans']:
        # --- Embeddings required ---
        embeddings = embed_model.encode(generations, convert_to_tensor=False, device=device)

        if cluster_mode == 'pca_only':
            for n_comp in range(1, max_components + 1):
                pca = PCA(n_components=n_comp)
                reduced = pca.fit_transform(embeddings)
                cluster_ids = np.digitize(reduced[:, 0],
                                          np.percentile(reduced[:, 0], [25, 50, 75]))
                all_cluster_assignments[f"pca{n_comp}"] = cluster_ids

        elif cluster_mode == 'kmeans_only':
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                cluster_ids = kmeans.fit_predict(embeddings)
                all_cluster_assignments[f"kmeans{k}"] = cluster_ids

        elif cluster_mode == 'pca_then_kmeans':
            for n_comp in range(1, max_components + 1):
                pca = PCA(n_components=n_comp)
                reduced = pca.fit_transform(embeddings)
                for k in range(1, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    cluster_ids = kmeans.fit_predict(reduced)
                    all_cluster_assignments[f"pca{n_comp}_k{k}"] = cluster_ids

    elif cluster_mode == 'semantic_clustering':
        if semantic_model is None or semantic_tokenizer is None:
            raise ValueError("semantic_model and semantic_tokenizer must be provided for semantic_clustering")
        cluster_ids = compute_semantic_similarity(sample, semantic_model, semantic_tokenizer, device=device)
        all_cluster_assignments['semantic'] = cluster_ids

    else:
        raise ValueError(f"Unknown cluster_mode: {cluster_mode}")

    return all_cluster_assignments


def compute_clusters_origin(sample, embed_model, device='cuda:0',
                     cluster_mode='pca_then_kmeans',
                     max_components=10, max_k=10):
    """
    Compute cluster assignments using different methods:
      - 'pca_only'         → PCA projections (1..max_components)
      - 'kmeans_only'      → KMeans (k=1..max_k)
      - 'pca_then_kmeans'  → PCA (n_components=1..max_components) then KMeans (k=1..max_k)

    Returns:
      dict {config_key: cluster_ids}
        e.g., 'pca3', 'kmeans5', or 'pca3_k5'
    """
    generations = sample['cleaned_generated_texts']
    embeddings = embed_model.encode(generations, convert_to_tensor=False, device=device)
    all_cluster_assignments = {}

    if cluster_mode == 'pca_only':
        for n_comp in range(1, max_components + 1):
            pca = PCA(n_components=n_comp)
            reduced = pca.fit_transform(embeddings)
            # Simple grouping by quantiles of first principal component
            cluster_ids = np.digitize(reduced[:, 0],
                                      np.percentile(reduced[:, 0], [25, 50, 75]))
            all_cluster_assignments[f"pca{n_comp}"] = cluster_ids

    elif cluster_mode == 'kmeans_only':
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            cluster_ids = kmeans.fit_predict(embeddings)
            all_cluster_assignments[f"kmeans{k}"] = cluster_ids

    elif cluster_mode == 'pca_then_kmeans':
        for n_comp in range(1, max_components + 1):
            pca = PCA(n_components=n_comp)
            reduced = pca.fit_transform(embeddings)
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                cluster_ids = kmeans.fit_predict(reduced)
                all_cluster_assignments[f"pca{n_comp}_k{k}"] = cluster_ids
    else:
        raise ValueError(f"Unknown cluster_mode: {cluster_mode}")

    return all_cluster_assignments


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