import sys
import os
sys.path.append(os.path.abspath("../predictor/"))
from bert_mdl import retrieve_model, extract_and_save_embeddings, compute_embs, get_saliency_map
import pandas as pd
import models
import utils
import torch
# include as demo vdj50 <-
def construct_model(p,n_feat,n_labels,device):
    if p['two_chains']:
        model = models.CNN3AB(n_feat,n_labels,kernel_sizes=p['kernel_sizes'],
                dos=p['dropouts'],pool=p['pool'])
    else:
        model = models.CNN3(n_feat,n_labels,kernel_sizes=p['kernel_sizes'],
                dos=p['dropouts'],pool=p['pool'])

    model.to(device)

    return model

if __name__=="__main__":

    # # load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    p = utils.get_parameter_dict(predict=True)
    n_labels = 51
    classification_head = utils.load_model(p['model_file'], p, p['num_features'], n_labels, device)

    model = retrieve_model()
    model.classification_head = classification_head
    model.to(device)


    get_saliency_map(model, data_f_n="data/vdj_human_uniques_long.csv")

    exit(1)
    # extract saliency maps
    print(model.classification_head)
    print()
    exit(1)


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
