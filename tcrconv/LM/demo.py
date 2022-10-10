import random
import pickle
import sys
import os
sys.path.append(os.path.abspath("../predictor/"))
from bert_mdl import retrieve_model, extract_and_save_embeddings, compute_embs, get_saliency_map
import pandas as pd
import models
import utils
import torch
# include as demo vdj50 <-


if __name__=="__main__":

    # # load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    p = utils.get_parameter_dict(predict=True)

    # TODO: e.g. statedict_vdjdb-b-small.pt has 21 labels. Change below if only-beta saliency maps are required
    n_labels = 18
    classification_head = utils.load_model(p['model_file'], p, p['num_features'], n_labels, device)

    model = retrieve_model()
    model.classification_head = classification_head
    model.to(device)

    # TODO add the correct epitope2label dictionary used to extract gradients of correct lbls (and not predicted ones)
    #  for now, I assign some random label for each epitope found in the vdj_human_unique50_aligned data
    #  use p['epitope_labels'] with a correct file (couldn't find the statedict_vdjdb-ab-large.pt epitope dictionary)
    data_ = pd.read_csv("data/vdj_human_unique50_aligned.csv")
    eps = []
    for e in data_['Epitope'].values:
        eps.extend(e.split(" "))
    epitope2lbl = {e:random.randint(0,17) for e in set(eps)}

    # TODO If only beta chains are required, use --chains B; else, use --chains AB.
    #  If true-epitope predictions are required, assign epitope2lbl the correct epitope2lbl dictionary for the loaded model
    #  If max-predicted predictions are required, assign epitope2lbl=None in the function below
    saliency_results = get_saliency_map(model, data_f_n="data/vdj_human_unique50_aligned.csv", chains=p['chains'], epitope2lbl=epitope2lbl)
    pickle.dump(saliency_results, open("some_save_file_name.bin", "wb"))



    extract_and_save_embeddings(model, data_f_n="data/cleaned_data.csv", sequence_col="long", cdr3_col="cdr3a", seqs_per_file=20000, emb_name='cdr3a_embs')
    print("finished")
    exit(123)

    # extract and save some embeddings
    extract_and_save_embeddings(model, data_f_n="data/cleaned_data.csv", sequence_col="long", cdr3_col="cdr3a", seqs_per_file=20000, emb_name='cdr3a_embs')
    print("finished")
    exit(123)
    # compute some more embeddings (in an "online" fashion)
    data = pd.read_csv("data/vdj_human_uniques_long.csv", sep=",")
    long, cdr3b = data["long"].values[:100], data["cdr3b"].values[:100]
    print(compute_embs(model, long)[0].shape, compute_embs(model, long)[-1].shape)
    print(compute_embs(model, long, cdr3b)[0].shape, compute_embs(model, long, cdr3b)[-1].shape)

    # load fine-tuned bert and compute some embeddings with the loaded model. A checkpoint model will be available after
    # fine-tuning the BERT model (use fine_tune_bert.py to tune the model on epitope prediction task or further
    # train it on th unsupervised task of predicting amino acids in sequences

    # checkpoint_path = "experiments/lightning_logs/version_example/checkpoints/checkpoint_example.ckpt"
    # model = model.load_from_checkpoint(checkpoint_path)
    # model.eval()
    # model.to(device)
    # print(compute_embs(model, long)[0].shape, compute_embs(model, long)[-1].shape)
    # print(compute_embs(model, long, cdr3b)[0].shape, compute_embs(model, long, cdr3b)[-1].shape)
