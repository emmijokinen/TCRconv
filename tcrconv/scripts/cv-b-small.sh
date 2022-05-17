#!/bin/bash -l
#SBATCH -t 0:05:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu[11-17]
#SBATCH --mem-per-cpu=3000
#SBATCH --array=0
#SBATCH -J cv_small
#SBATCH -o cv_small.out

set -e
source ../tcrconv-env/bin/activate

python3 predictor/run_tcrconv.py --name b_small \
--dataset training_data/vdjdb-b-small.csv \
--mode cv \
--h_cdr31 CDR3B --h_long1 LongB \
--folds training_data/folds_vdjdb-b-small.npy \
--fold_num $SLURM_ARRAY_TASK_ID \
--epitope_labels training_data/unique_epitopes_vdjdb-b-small.npy \
--chains B --embedfile1 embeddings/bert_vdjdb-b-small_0.bin \
