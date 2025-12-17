# RDS

**Distance Is All You Need: Radial Dispersion for Uncertainty Estimation in Large Language Models** 

- Preprint: [arXiv:2512.04351](https://arxiv.org/abs/2512.04351)

---

## 1. Environment Setup

Create a conda environment from the provided YAML file:

```bash
conda env create -f environment.yaml
conda activate <env_name>
```
Before running any scripts, make sure to update file paths in `config.py` according to your local directory structure.

## 2. Generation

```bash
python generation.py --model {model_name} --dataset {dataset_name} --n_samples 10 --fraction_of_data_to_use 1
```

#### Parameters: 
* `--model`: Model identifier (e.g., `qwen2.5-7b`, `llama3.1-8b`)
* `--dataset`: Dataset name
* `--n_samples`: Number of generations per input
* `--fraction_of_data_to_use`: Fraction of the dataset to use (1 = full dataset)

## 3. Hallucination Detection Performance

```bash
python uncertainty.py --model {model_name} --dataset {dataset_name} --n_samples 10 --semantic_baselines True
```

#### Optional Flags: 
* `--semantic_baselines`: Enables semantic uncertainty baselines: SE (Semantic Entropy), Deg (Degree Matrix), SD (Semantic Density)

## 4. Best-of-N Performance

```bash
python ranking.py --model {model_name} --dataset {dataset_name} --n_samples 10 --self_certainty True
```

#### Optional Flags:
* `--self_certainty`: Enables CE baseline for ranking.

## 📓 Example Workflow
```bash
bash run.sh
```