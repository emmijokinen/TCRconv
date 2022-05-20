#!/bin/bash -l
#SBATCH -t 0:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH -J train_blarge
#SBATCH -o train_blarge.out

set -e
source ../tcrconv-env/bin/activate

python3 predictor/run_tcrconv.py --name vdjdb-b-large \
--dataset training_data/vdjdb-b-large.csv \
--epitope_labels training_data/unique_epitopes_vdjdb-b-large.npy \
--mode train \
--chains B --h_cdr31 CDR3B --h_long1 LongB \
--embedfile1 embeddings/bert_vdjdb-b-large_0.bin \
--model_folder models
