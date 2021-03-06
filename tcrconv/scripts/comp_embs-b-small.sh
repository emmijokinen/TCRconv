#!/bin/bash -l
#SBATCH -t 0:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10000
#SBATCH -J emb_bsmall
#SBATCH -o emb_bsmall.out

set -e
source ../tcrconv-env/bin/activate

python LM/compute_embeddings.py \
--name embeddings/bert_vdjdb-b-small \
--dataset training_data/vdjdb-b-small.csv \
--h_cdr3 CDR3B --h_long LongB --delimiter ,