#!/bin/bash -l
#SBATCH -t 0:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH -J train_ablarge
#SBATCH -o train_ablarge.out

set -e
source ../tcrconv-env/bin/activate

python3 predictor/run_tcrconv.py --name vdjdb-ab-large \
--dataset training_data/vdjdb-ab-large.csv \
--epitope_labels training_data/unique_epitopes_vdjdb-ab-large.npy \
--mode train \
--chains AB \
--h_cdr31 CDR3A --h_long1 LongA \
--h_cdr32 CDR3B --h_long2 LongB \
--embedfile1 embeddings/bert_vdjdb-ab-large_a_0.bin \
--embedfile2 embeddings/bert_vdjdb-ab-large_b_0.bin \
--model_folder models
