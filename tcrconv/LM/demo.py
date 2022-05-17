from bert_mdl import retrieve_model, extract_and_save_embeddings, compute_embs
import pandas as pd
import torch

# include as demo vdj50 <-

if __name__=="__main__":

    # # load the model
    model = retrieve_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # extract and save some embeddings
    extract_and_save_embeddings(model, data_f_n="data/cleaned_data.csv", sequence_col="long", cdr3_col="cdr3a", seqs_per_file=20000, emb_name='cdr3a_embs')
    print("finished")
    exit(123)
    # compute some more embeddings (in an "online" fashion)
    data = pd.read_csv("data/vdj_human_unique_longs.csv", sep=",")
    long, cdr3b = data["long"].values[:100], data["cdr3b"].values[:100]
    print(compute_embs(model, long)[0].shape, compute_embs(model, long)[-1].shape)
    print(compute_embs(model, long, cdr3b)[0].shape, compute_embs(model, long, cdr3b)[-1].shape)

    # load fine-tuned bert and compute some embeddings with the loaded model. A checkpoint model will be available after
    # fine-tuning the BERT model
    best_checkpoint_path = "experiments/lightning_logs/version_17-9-2021--13-39-35/checkpoints/epoch=8-val_loss=1.51-val_acc=0.96.ckpt"
    model = model.load_from_checkpoint(best_checkpoint_path)
    model.eval()
    model.to(device)
    print(compute_embs(model, long)[0].shape, compute_embs(model, long)[-1].shape)
    print(compute_embs(model, long, cdr3b)[0].shape, compute_embs(model, long, cdr3b)[-1].shape)
