#!/bin/bash -l
#SBATCH -t 0:05:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3000
#SBATCH -J train_bsmall
#SBATCH -o train_bsmall.out

set -e
source ../tcrconv-env/bin/activate

python3 predictor/run_tcrconv.py --name vdjdb-b-small \
--dataset training_data/vdjdb-b-small.csv \
--epitope_labels training_data/unique_epitopes_vdjdb-b-small.npy \
--mode train \
--chains B --h_cdr31 CDR3B --h_long1 LongB \
--embedfile1 embeddings/bert_vdjdb-b-small_0.bin \
--model_folder models
