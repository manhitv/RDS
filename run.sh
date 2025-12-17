python generation.py --model falcon3-7b --dataset gpqa --n_samples 10 --fraction_of_data_to_use 1

python uncertainty.py --model falcon3-7b --dataset gpqa --n_samples 10 --semantic_baselines True

python ranking.py --model falcon3-7b --dataset gpqa --n_samples 10 --self_certainty True