#!/bin/bash -l
#SBATCH -t 0:05:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3000
#SBATCH --array=1
#SBATCH -J pred_bsmall
#SBATCH -o pred_bsmall.out

set -e
source ../tcrconv-env/bin/activate

python predictor/pred_tcrconv.py \
--dataset training_data/vdjdb-b-small.csv \
--mode prediction \
--h_cdr31 CDR3B --h_long1 LongB \
--model_file models/statedict_b_small_0.pt \
--epitope_labels training_data/unique_epitopes_vdjdb-b-small.npy \
--chains B \
--embedfile1 embeddings/bert_vdjdb-b-small_0.bin \
--predfile outputs/preds-b-small.csv --batch_size 256 \
--additional_columns CDR3B Subject
