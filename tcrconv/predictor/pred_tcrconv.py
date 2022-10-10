
import torch
import numpy as np
import csv
import utils
from pandas import read_csv, DataFrame
from tcrconv.preprocessing.prep import get_protseqs_ntseqs,determine_tcr_seq_nt,determine_tcr_seq_vj
from tcrconv.LM.bert_mdl import retrieve_model, compute_embs

# Read arguments
p = utils.get_parameter_dict(predict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = read_csv(p['dataset'],delimiter=p['delimiter'],dtype=str,keep_default_na=False)
epis_u = np.load(p['epitope_labels'])
n_labels=len(epis_u)
n_chains=len(p['chains'])

resfile = p['predfile']
with open(resfile,'w') as f:
    f.write(p['delimiter'].join(['TCR'+c for c in p['chains']]
            +p['additional_columns']+list(epis_u))+'\n')

# Load TCRconv model
model = utils.load_model(p['model_file'],p,p['num_features'],n_labels,device)
# embeddings / embedding-models
if p['use_LM']:
    LM=retrieve_model().to(device)

# Get requested sequences for genes
geneseqs={}
for ic,chain in enumerate(p['chains']):
    c=str(ic+1)
    if p['input_type']=='cdr3+nt':
        geneseqs['protV'+c],geneseqs['protJ'+c],geneseqs['ntV'+c],geneseqs['ntJ'+c] = \
                get_protseqs_ntseqs(chain=chain)
    elif p['input_type']=='cdr3+vj':
        geneseqs['protV'+c],geneseqs['protJ'+c],_,_ = get_protseqs_ntseqs(chain=chain)

I=[]
ts_all = [[] for c in p['chains']] # separate list for each chain
icount,i0 = 0,0
imax=len(data)-1

for i in range(len(data)):

    if p['input_type']=='cdr3+nt':
        tcr12 = []
        for ic in range(n_chains):
            c=str(ic+1)
            t,_,_ = determine_tcr_seq_nt(data[p['h_nt'+c]][i],data[p['h_cdr3'+c]][i],geneseqs['protV'+c],
                    geneseqs['protJ'+c],geneseqs['ntV'+c],geneseqs['ntJ'+c],guess01=p['guess_allele01'])
            tcr12.append(t)
    elif p['input_type']=='cdr3+vj':
        tcr12 = []
        for ic in range(n_chains):
            c=str(ic+1)
            t = determine_tcr_seq_vj(data[p['h_cdr3'+c]][i],data[p['h_v'+c]][i],data[p['h_j'+c]][i],
                geneseqs['protV'+c],geneseqs['protJ'+c],guess01=p['guess_allele01'])
            tcr12.append(t)
    elif p['input_type']=='cdr3':
        tcr12 = [data[p['h_cdr3'+str(ic+1)]][i] for ic in range(n_chains)]
    else: # tcr+cdr3 / tcr
        tcr12 = [data[p['h_long'+str(ic+1)]][i] for ic in range(n_chains)]


    # Check if (either) sequence is empty
    if np.any([t=='' for t in tcr12]):
        I.append(False)
        for ic in range(n_chains):
            ts_all[ic].append(tcr12[ic])
    else:
        I.append(True)
        for ic in range(n_chains):
            ts_all[ic].append(tcr12[ic])
        icount+=1


    if icount==p['batch_size'] or i==imax:
        ts_all =[np.array(t) for t in ts_all]
        if icount>0:
            I= np.array(I,dtype=bool)
            if p['use_LM']: # If LM is used, compute embeddings
                cdr3s = data[p['h_cdr31']][i0:i+1][I].values
                print('computing embeddings: {:d}-{:d}/{:d}'.format(i0,i,imax))
                if p['embedtype']=='cdr3+context':
                    embeddings1 = compute_embs(LM, ts_all[0][I], cdr3s)
                    embeddings1 = utils.stack_embeddings(embeddings1,device,p['append_oh'],cdr3s)
                else:
                    embeddings1 = compute_embs(LM, ts_all[0][I], None)
                    embeddings1 = utils.stack_embeddings(embeddings1,device,p['append_oh'])

                if p['two_chains']:
                    cdr3s = data[p['h_cdr32']][i0:i+1][I].values
                    if p['embedtype']=='cdr3+context':
                        embeddings2 = compute_embs(LM,ts_all[1][I], cdr3s)
                        embeddings2 = utils.stack_embeddings(embeddings2,device,p['append_oh'],cdr3s)
                    else:
                        embeddings2 = compute_embs(LM, ts_all[1][I], None)
                        embeddings2 = utils.stack_embeddings(embeddings2,device,p['append_oh'])

            else: # 1-2 embedding dictionaries are used

                #print(ts_all[0][I])
                cdr3s = data[p['h_cdr31']][i0:i+1][I].values
                cdr3max = utils.maxlen(cdr3s)
                embeddings1 = utils.get_embeddings(utils.get_embedding_dict(p['embedfile1']),
                            cdr3s, ts_all[0][I], cdr3max, device, p['append_oh'])
                if p['two_chains']:
                    cdr3s = data[p['h_cdr32']][i0:i+1][I].values
                    cdr3max = utils.maxlen(cdr3s)
                    embeddings2 = utils.get_embeddings(utils.get_embedding_dict(p['embedfile2']),
                            cdr3s,ts_all[1][I], cdr3max, device, p['append_oh'])

            # Predictions
            if p['two_chains']:
                output = model(embeddings1,embeddings2).detach().cpu().numpy()
            else:
                output = model(embeddings1).detach().cpu().numpy()
            output = utils.toprob(output)

            pred_ar = np.ones((len(I),n_labels),dtype=float)*np.nan
            pred_ar[I,:]= output

        else: # No proper sequences were found, add fillers
            pred_ar = np.ones((len(I),n_labels),dtype=float)*np.nan

        # append results to result file
        df = DataFrame(np.concatenate([np.expand_dims(ts_all[i],1) for i in range(len(ts_all))] \
            +[np.expand_dims(data[col].values[i0:i+1],1) for col in p['additional_columns']] \
            +[np.round(pred_ar,p['decimals'])],axis=1))
        df.to_csv(resfile,sep=p['delimiter'],mode='a',header=False,index=False)

        I = []
        ts_all = [[] for c in p['chains']]
        icount = 0
        i0 = i+1
