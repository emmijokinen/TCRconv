#!/bin/bash -l
#SBATCH -t 04:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu[11-17]
#SBATCH --mem-per-cpu=8000
#SBATCH -J sal_abb
#SBATCH -o slurmout/saliency_ab-b.out

set -e
source ../tcrconv-env/bin/activate

python LM/saliency.py \
--dataset training_data/vdjdb-ab-large.csv \
--mode prediction \
--h_cdr31 CDR3B --h_long1 LongB \
--model_file models/statedict_vdjdb-ab-large-b.pt \
--epitope_labels training_data/unique_epitopes_vdjdb-ab-large.npy \
--chains B \
--predfile outputs/saliency_ab-b.bin \
--batch_size 32 \
--use_LM True
