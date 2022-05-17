
import models
import torch
from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR

from re import search as re_search
from os import path
import csv
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter,MetavarTypeHelpFormatter
from pandas import read_csv

##### PARSE ARGUMENTS #############################################################
class CustomFormatter(ArgumentDefaultsHelpFormatter,MetavarTypeHelpFormatter):
    pass

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parameter_dict(predict=False):
    parser = ArgumentParser(description='TCRconv parameters',formatter_class=CustomFormatter)

    p_model = parser.add_argument_group('Model parameters','Define parameters for the predictor. When an existing model is used for prediction (with pred_tcrconv.py), the model parameters must be the same as what were used when training the model (as only state_dict is saved and loaded).')
    p_result = parser.add_argument_group('Result parameters','Define what results are saved and where.')
    p_data = parser.add_argument_group('Data parameters','Define what data to use')

    # Data and folds

    p_data.add_argument('--dataset', type=str, required=True,
                        help='filename of the used dataset')
    p_data.add_argument('--chains', type=str, required=True,choices=['A','B','AB','BA'],
                        help='Names for the used chains, first chain 1 and then chain 2')
    p_data.add_argument('--epitope_labels', type=str, required=True,
                        help='filename of the numpy array with epitope labels. Results will be given in corresponding order')
    p_data.add_argument('--embedtype',type=str, default='cdr3+context',choices=['cdr3+context','tcr','cdr3'],
                        help='How the embedding is constructed, cdr3+context: embeddings are based on the CDR3 with full TCR context, tcr: embeddings use the complete TCR sequence,cdr3: embeddings are based on only the cdr3 sequence without additional context .')
    # First chain
    p_data.add_argument('--h_cdr31',type=str, default='CDR3B',
                        help='Column name for CDR3 of chain 1 in dataset file')
    p_data.add_argument('--h_long1',type=str, default='LongB',
                        help='Column name for Long TCR-sequence of chain 1 in dataset file')
    p_data.add_argument('--embedfile1', type=str, default='None',
                        help='filename for embeddings for chain 1')
    # Second chain (optional)
    p_data.add_argument('--h_cdr32',type=str, default='None',
                        help='Column name for CDR3 of chain 2 in dataset file')
    p_data.add_argument('--h_long2',type=str, default='None',
                        help='Column name for Long TCR-sequence of chain 2 in dataset file')
    p_data.add_argument('--embedfile2', type=str, default='None',
                        help='filename for embeddings for chain 2')
    p_data.add_argument('--delimiter', type=str, default=',',
                        help='Delimiter used in dataset file. For tab ("\\t"), give "tab".')

    if not predict: # Parameters just for training'
        p_usage = parser.add_argument_group('Model usage', 'Crossvalidation or training')
        p_train = parser.add_argument_group('Training parameters','Define parameters for training and for the optimizer')
        p_usage.add_argument('--mode', type=str, required=True, choices=['cv','train'],
                            help='cv: cross-validation, folds and fold_number are required. \
                            train: train a model using all data in dataset file. Use pred_tcrconv.py for prediction.')
        p_model.add_argument('--name', type=str, required=True,
                        help='name for the model. Will be used if the model or results are saved.')
        p_data.add_argument('--folds', type=str, default='None',
                            help='filename of the used dataset')
        p_data.add_argument('--fold_num',type=int, default=0,
                            help='Number of the fold to be used. Must be present in folds')
        p_data.add_argument('--h_epitope',type=str, default='Epitope',
                            help='Column name for epitopes in dataset file')

        # Training parameters: general
        p_train.add_argument('--batch_size',type=int, default=512,
                            help='Batch size used during training. If a batch size smaller than the default 512 is used during training, it\'s good to increase the number of iterations accordingly.')
        p_train.add_argument('--betas',type=float, nargs=2, default=[0.9,0.999],
                            help='Beta coefficients for Adam optimizer for computing running averages of gradient and its square')
        p_train.add_argument('--use_pos_weight',type=str2bool, default=True,
                            help='If true, positive labels are given weights by their frequency in the training data so that the training does is not dominated by more abundant labels.')
        p_train.add_argument('--dropouts',type=float, nargs=2, default=[0.1,0.1],
                            help='Values for the first and second dropouts.')

        # Training parameters: before swa
        p_train.add_argument('--iters_adam',type=int, default=2500,
                            help='Number of iterations to be used before SWA')
        p_train.add_argument('--lr_conv',type=float, default=0.0002,
                            help='Learning rate for convolutional unit.')
        p_train.add_argument('--lr_linear',type=float, default=0.01,
                            help='Learning rate for linear unit.')
        p_train.add_argument('--T_anneal',type=int, default=3000,
                            help='Maximum number of iterations for CosineAnnealingLR before SWA')

        # Training parameters: swa
        p_train.add_argument('--iters_swa',type=int, default=500,
                            help='Number of iterations to be used before SWA')
        p_train.add_argument('--anneal_strategy',type=str, default='cos',choices=['cos','linear'],
                            help='Annealing strategy for SWALR')
        p_train.add_argument('--lr_swa',type=float, default=0.0001,
                            help='Learning rate for swa.')
        p_train.add_argument('--T_anneal_swa',type=int, default=300,
                            help='Maximum number of iterations for CosineAnnealingLR during SWA')

        # Results
        p_result.add_argument('--resultfile',type=str, default='outputs/results.tsv',
                            help='Filename for saving results. Output from a fold will be written on one row.')
        p_result.add_argument('--print_every',type=int, default=100,
                            help='how many iterations between progress tracking')
        p_result.add_argument('--lossfile',type=str, default='outputs/loss_train.tsv',
                            help='Filename for saving tracked loss for training data. Output from a fold will be written on one row.')
        p_result.add_argument('--lossfile_test',type=str, default='outputs/loss_test.tsv',
                            help='Filename for saving tracked loss for test data. Outputs from a fold will be written on three rows: loss, AUROC and AP scores.')
        p_result.add_argument('--model_folder',type=str,default='models',help='Folder where the model \
                            will be saved, models are named as <name>_<fold_num>. Give None if you \
                            do not want to save the model.')
        p_result.add_argument('--params_to_print',type=str,nargs='*',default=['name','fold_num'],
                            help='Give list of parameters to be added in result tables.')

    else: # parameters just for predicting
        p_usage = parser.add_argument_group('Model usage', 'Prediction')
        p_usage.add_argument('--mode', type=str, default='prediction', choices=['prediction'],
                            help='pred_tcrconv can only be used for prediction. Use train_tcrconv.py for training.')
        p_data.add_argument( '--h_v1', type=str, default='none', \
                                help='Column name for V-gene of chain 1 in dataset file. \
                                Required if input_type is cdr3+vj.')
        p_data.add_argument( '--h_j1', type=str, default='none', \
                                help='Column name for J-gene of chain 1 in dataset file \
                                Required if input_type is cdr3+vj.')
        p_data.add_argument( '--h_v2', type=str, default='none', \
                                help='Column name for V-gene of chain 2 in dataset file \
                                Required if input_type is cdr3+vj and two chains are used.')
        p_data.add_argument( '--h_j2', type=str, default='none', \
                                help='Column name for J-gene of chain 2 in dataset file \
                                Required if input_type is cdr3+vj and two chains are used.')
        p_data.add_argument( '--h_nt1', type=str, default='none', \
                                help='Column name for nucleotide seq of chain 1 in dataset file \
                                Required if input_type is cdr3+nt.')
        p_data.add_argument( '--h_nt2', type=str, default='none', \
                                help='Column name for nucleotide seq of chain 2 in dataset file \
                                Required if input_type is cdr3+nt and two chains are used.')
        p_data.add_argument( '--guess_allele01', type=str2bool, default=True, \
                                help='Applicable if input_type is cdr3+vj or cdr3+nt. When \
                                determining the TCR amino-acid sequence based on V- and J-genes or \
                                the nucleotide sequence, if True when ever a V- or J-gene is known \
                                but its allele is not, guess the allele to be 01.')


        p_model.add_argument('--model_file', type=str, required=True,
                            help='filename for model.')
        p_model.add_argument('--LM_file', type=str, default='None',
                            help='filename for LM. LM or embeddings required with mode predict.')
        p_model.add_argument('--num_features', type=int,default=1024,
                            help='Number of features in embeddings. If a bert model is used, this should be 1024. If a bert model with one-hot encoding is used, this should be 1045 (1024+21)')
        p_data.add_argument('--input_type', type=str, default='tcr+cdr3',
                            choices=['tcr+cdr3','cdr3+vj','cdr3+nt','cdr3','tcr'],
                            help='Type of the input sequences: tcr+cdr3 (embeddings are based directly on these, cdr3+context), cdr3+vj / cdr3+nt (tcr sequence is first determined based on these), cdr3 (embeddings are based only on cdr3 sequence, no context), tcr: (embeddings are based on the full tcr sequence).')
        p_result.add_argument('--predfile', type=str, required=True,
                            help='Name for the file where predictions will be saved')
        p_result.add_argument('--additional_columns', type=str, nargs='*', default=[],
                            help='Names of the columns that will be added to the predfile in addition to the used TCR-seq and predictions.')
        p_result.add_argument('--decimals', type=int, default=4,
                            help='How many decimals are saved for the predictions')
        p_batch = parser.add_argument_group('Batch size')
        p_batch.add_argument('--batch_size',type=int, default=512,
                            help='Defines for how many sequences embeddings and predictions are computed at a time. If an LM is used instead of precomputed embeddings, using larger batch_size can be quicker but requires more memory.')

    # Opt to use binary instead of multilabel classification
    p_data.add_argument('--binary', type=str2bool, default=False,
                        help='If True, binary classification is performed instead of multilabel classification. Epitope label ')
    p_data.add_argument('--binary_label', type=str,
                        help='If binary is True, the TCRs recognizing the selected epitope are considered as positive and other TCRs as negatives. binary_label must be in epitope_labels')

    # Model parameters
    p_model.add_argument('--kernel_sizes', type=int, nargs=5, default=[5,9,15,21,3],
                        help='kernelsizes for Conv1d-layers')
    p_model.add_argument('--append_oh', type=str2bool, default=False,
                        help='If onehot encoding will be concateated to the used embeddings')
    p_model.add_argument('--pool',type=str, default='max',choices=['max','sum','avg'],
                        help='Type of pooling to be used: max/sum/avg')

    p = parser.parse_args()

    # Let's use this as a dictionary
    p = p.__dict__

    p['two_chains']=len(p['chains'])>1
    p['delimiter'] = '\t' if p['delimiter']=='tab' else p['delimiter']

    if not predict: # Checks for cv and train modes
        if p['mode']=='cv' and p['folds'].lower()=='none':
            parser.error('If cross-validation is used, folds are required.')

        if p['mode']=='train':
            if p['lossfile_test'].lower()!='none' or p['resultfile'].lower()!='none':
                p['lossfile_test']='None'
                p['resultfile']='None'
                print('NOTE: In train mode test results cannot be computed. '+ \
                    'lossfile_test and resultfile have been set to None.')
            if p['folds'].lower()!='none':
                print('NOTE: In train mode all data is used for training the model. folds are ignored.')

        if p['two_chains'] and ((p['h_cdr32'].lower()=='none' and p['h_long2'].lower()=='none') or \
                p['embedfile2'].lower()=='none'):
            parser.error('If two chains are used, give column name for CDR3s or long sequences'+ \
            ', and embedfile for the 2nd chain.')

        if p['embedtype']=='cdr3':
            p['h_long1'],p['h_long2'] = p['h_cdr31'],p['h_cdr32']
        if p['embedtype']=='tcr':
             p['h_cdr31'],p['h_cdr32'] = p['h_long1'],p['h_long2']

        if p['embedtype']=='cdr3+context' and \
            ((p['h_cdr31'].lower()=='none' or p['h_long1'].lower()=='none') or \
            (p['two_chains'] and (p['h_cdr32'].lower()=='none' or p['h_long2'].lower()=='none'))):
            parser.error('If embedtype is cdr3+context, headers for both cdr3 and tcr must be given (for each chain).')

        p['save_intermediate'] = p['lossfile'].lower()!='none' or p['lossfile_test'].lower()!='none'
        # Parameter names and their values for result files
        p['table']=(p['params_to_print'],[p[h] for h in p['params_to_print']])

    else: # Checks for prediction mode
        # check that embedtype and input_type are compatible
        if p['input_type'] in ['cdr3','tcr'] and p['embedtype'] != p['input_type']:
            parser.error('If input_type is cdr3 or tcr, the embedtype must be the same')
        if p['input_type'] =='cdr3+vj' and (p['h_v1'].lower() == 'none' or p['h_j1'].lower()=='none'):
            parser.error('If input_type is cdr3+vj, V- and J-genes are required (for each chain).')
        if p['input_type'] =='cdr3+bt' and p['h_nt1'].lower() == 'none':
           parser.error('If input_type is cdr3+nt, nucleotide seq is required (for each chain).')

        if p['two_chains'] and p['h_cdr32'].lower()=='none' and p['h_long2'].lower()=='none':
            parser.error('If two chains are used, give column name for CDR3s or long sequences of the 2nd chain.')
        if (p['LM_file'].lower()=='none') == (p['embedfile1'].lower()=='none'):
            parser.error('Give either an LM or embedding file (for each chain).')

        # Dropouts are not used when the model is evaluated. These are just fillers
        p['dropouts']=[0,0]

    return p


##### TINY HELPERS #############################################################

def letterToIndex(letter):
    return 'ARNDCEQGHILKMFPSTWYVX'.index(letter)

def toprob(x,binary=False):
    if binary: # for softmax activation use exp
        return np.exp(x)
    else:   # sigmoid for multilabel classification
        return 1/(1 + np.exp(-x))

def maxlen(seqlist):
    return np.max([len(s) for s in seqlist])

def lineToTensor(line, device,n_letters=21,linelen=None):
    linelen= len(line) if linelen is None else linelen
    tensor = torch.zeros(n_letters, linelen, device=device)
    for li, letter in enumerate(line):
        tensor[letterToIndex(letter)][li] = 1
    return tensor

def lineToTensor2(line, n_letters=21,linelen=None):
    linelen= len(line) if linelen is None else linelen
    tensor = torch.zeros(1, n_letters, linelen)
    for li, letter in enumerate(line):
        tensor[0][letterToIndex(letter)][li] = 1
    return tensor

def lineToArray(line, n_letters=21,linelen=None):
    linelen= len(line) if linelen is None else linelen
    ar = np.zeros(n_letters, linelen)
    for li, letter in enumerate(line):
        ar[letterToIndex(letter)][li] = 1
    return ar

##### HANDLE DATA ##############################################################
def create_resultfiles(p,n_categories):
    # Create requested loss and result files if they don't exist yet
    # based on parameter dictionary p
    if p['lossfile']!='None' and not path.isfile(p['lossfile']):
        with open(p['lossfile'],'w') as f:
            writer=csv.writer(f,delimiter='\t')
            writer.writerow(p['table'][0]+['Output type'])
    if p['lossfile_test']!='None' and not path.isfile(p['lossfile_test']):
        with open(p['lossfile_test'],'w') as f:
            writer=csv.writer(f,delimiter='\t')
            writer.writerow(p['table'][0]+['Output type'])
    if p['resultfile']!='None' and not path.isfile(p['resultfile']):
        with open(p['resultfile'],'w') as f:
            writer=csv.writer(f,delimiter='\t')
            header =p['table'][0]
            header += ['AUROC_'+str(i) for i in range(n_categories['test'])]+['AUROC_micro','AUROC_macro']
            header += ['AP_'+str(i) for i in range(n_categories['test'])]+['AP_micro','AP_macro']
            writer.writerow(header)

def save_results(y_score, labels, p, n):
    auroc,ap = get_aurocs_aps(labels,y_score)
    auroclist = [auroc[i] for i in range(n)]+[auroc['micro'],auroc['macro']]
    aplist = [ap[i] for i in range(n)]+[ap['micro'],ap['macro']]

    with open(p['resultfile'],'a') as f:
        writer=csv.writer(f,delimiter=p['delimiter'])
        writer.writerow(p['table'][1]+auroclist+aplist)


def file2dict(filename,key_fields,store_fields,delimiter='\t'):
    """Read file to a dictionary.
    key_fields: fields to be used as keys
    store_fields: fields to be saved as a list
    delimiter: delimiter used in the given file."""
    dictionary={}
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile,delimiter=delimiter)
        for row in reader:
            keys = [row[k] for k in key_fields]
            store= [row[s] for s in store_fields]

            sub_dict = dictionary
            for key in keys[:-1]:
                if key not in sub_dict:
                    sub_dict[key] = {}
                sub_dict = sub_dict[key]
            key = keys[-1]
            if key not in sub_dict:
                sub_dict[key] = []
            sub_dict[key].append(store)
    return dictionary

class tcrDataset(Dataset):
    """Dataset of embedded TCRs"""

    def __init__(self, embeddings, labels):
        """
        Args:
            embeddings: torch tensor of embeddings (padded)
            labels: torch tensor of labels
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        embeddings = self.embeddings[idx]
        labels = self.labels[idx]
        sample = {'embeddings': embeddings, 'labels': labels, 'idx': idx}

        return sample

class tcrDatasetAB(Dataset):
    """Dataset of embedded TCRs."""

    def __init__(self, embeddings1, embeddings2, labels):
        """
        Args:
            embeddings1: torch tensor of embeddings (padded) for seqtype 1 (e.g. 'TCRb')
            embeddings2: torch tensor of embeddings (padded) for seqtype 2 (e.g. TCRa)
            labels: torch tensor of labels
        """
        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.labels = labels

    def __len__(self):
        return len(self.embeddings1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        embeddings1 = self.embeddings1[idx]
        embeddings2 = self.embeddings2[idx]
        labels = self.labels[idx]
        sample = {'embeddings1': embeddings1,'embeddings2': embeddings2, 'labels': labels, 'idx': idx}

        return sample


def get_labels(epis,epis_u_fixed=None, fixedplus=False):
    """ epis: list or array of epitopes for which the labels will be extracted
        epis_u_fixed: list or array (of length l) of unique epitopes that will define the first l labels
        fixedplus: If True and epis_u_fixed is not None, define labels from additional epitopes in epis
            that don't occur in epis_u_fixed. These will be added to end of laels in alphabetical order
    """

    if epis_u_fixed is not None and ~fixedplus:
        epis_u = epis_u_fixed
    else:
        epis_u0 = np.unique(epis)
        epis_u=[]
        for epis_i in epis_u0:
            for e in epis_i.split(' '):
                epis_u.append(e)
        epis_u=np.unique(epis_u)
        if fixedplus:
            epis_u=np.concatenate((epis_u_fixed,list(filter(lambda epi: epi not in epis_u_fixed,epis_u))))

    labels_ar=np.zeros((len(epis),len(epis_u)),dtype=bool)
    for i,epis_i in enumerate(epis):
        for e in epis_i.split(' '):
            ind = np.nonzero(epis_u==e)[0]
            labels_ar[i][ind]=1

    return epis_u,labels_ar

def get_embedding_dict(embedfile):
    # Get correct embedding dictionary
    tcr_emb_dict = pickle.load(open(embedfile,'rb'))

    return tcr_emb_dict


def stack_embeddings(embedding_list,device,append_onehot=False,cdr3s=None):
    """Given a list of embeddings, stack them into one tensor. If append_onehot is True, cdr3s (or tcrs) are required and their one-hot encodings are concatenated to the embeddings."""
    nfeat=embedding_list[0].shape[0]
    cdr3max = np.max([e.shape[1] for e in embedding_list])
    if append_onehot:
        n_letters=21 # Number of letters in alphabet (of letterToIndex)
        embeddings=torch.zeros((len(embedding_list),n_letters+nfeat,cdr3max),dtype=torch.float32,device=device)
        for i,cdr3 in enumerate(cdr3s):
            li=len(cdr3)
            embeddings[i,:n_letters,:li]=lineToTensor(cdr3,linelen=li,device=device)
            embeddings[i,n_letters:,:li]=torch.as_tensor(embedding_list[i],device=device)
    else:
        embeddings=torch.zeros((len(embedding_list),nfeat,cdr3max),dtype=torch.float32,device=device)
        for i,emb in enumerate(embedding_list):
            embeddings[i,:,:emb.shape[1]]=torch.as_tensor(emb,device=device)

    # shape : n_tcrs x n_features x maxlen
    return embeddings

def get_embeddings(tcr_emb_dict,cdr3s,longs,cdr3max,device,append_onehot=False):
    nfeat=tcr_emb_dict[longs[0]].shape[0]
    if append_onehot:
        n_letters=21 # Number of letters in alphabet (of letterToIndex)
        embeddings=torch.zeros((len(cdr3s),n_letters+nfeat,cdr3max),dtype=torch.float32,device=device)
        for i,(cdr3,tcr) in enumerate(zip(cdr3s,longs)):
            li=len(cdr3)
            embeddings[i,:n_letters,:li]=lineToTensor(cdr3,linelen=li,device=device)
            embeddings[i,n_letters:,:li]=torch.as_tensor(tcr_emb_dict[tcr],device=device)
    else:
        embeddings=torch.zeros((len(cdr3s),nfeat,cdr3max),dtype=torch.float32,device=device)
        for i,(cdr3,tcr) in enumerate(zip(cdr3s,longs)):
            embeddings[i,:,:len(cdr3)]=torch.as_tensor(tcr_emb_dict[tcr],device=device)

    # shape : n_tcrs x n_features x maxlen
    return embeddings

def get_seqdata(usetype,p,h_epitope,h_cdr3,h_long):

    epis_u= np.load(p['epitope_labels'])
    data = read_csv(p['dataset'],delimiter=p['delimiter'])
    epis,cdr3s,longs = data[h_epitope].values, data[h_cdr3].values, data[h_long].values

    # if p['mode']=='train', use everything, else selected sequences for training/testing
    if p['mode']=='cv':
        folds = np.load(p['folds'])
        if 'test' in usetype: # get data for testing
            I = folds == p['fold_num']
        elif usetype == 'train':
            I = folds != p['fold_num']
        epis,cdr3s,longs = epis[I],cdr3s[I],longs[I]

    epis_u,labels_ar = get_labels(epis,epis_u)

    return cdr3s,longs,labels_ar,epis_u

def get_dataloaders(p,device,usetypes=['train','test']):
    """ Get DataLoaders for given usetypes and the given fold number.
    regions are only needed if Graph neural networks are used"""

    tcr_emb_dict = get_embedding_dict(p['embedfile1'])
    if p['two_chains']:
        tcr_emb_dict2 = get_embedding_dict(p['embedfile2'])
    else:
        tcr_embdict2 = None

    # Get dataloaders for selected usetypes. Shuffle only for use type train.
    loader, n_categories = {}, {}
    for usetype in usetypes:
        # First chain
        cdr3s, longs,labels_ar,epis_u = get_seqdata(usetype, p,p['h_epitope'],p['h_cdr31'],p['h_long1'])
        if p['binary']: # If binary classification, get labels only for the spesified class and its negative
            ind = list(epis_u).index(p['binary_label'])
            labels_ar = np.concatenate((labels_ar[:,ind:ind+1], 1-labels_ar[:,ind:ind+1]), axis=1)
        labels_ar = torch.as_tensor(labels_ar,dtype=torch.float32,device=device)

        if 'cdr3'==p['embedtype']: # Use CDR3 sequence without context
            embeddings = get_embeddings(tcr_emb_dict,cdr3s,cdr3s,maxlen(cdr3s),append_onehot=p['append_oh'],device=device)
        elif 'tcr' ==p['embedtype']: # Use full TCR sequence
            embeddings = get_embeddings(tcr_emb_dict,longs,longs,maxlen(longs),append_onehot=p['append_oh'],device=device)
        else: #cdr3+context: Use CDR3 sequence with full context (recommended)
            embeddings = get_embeddings(tcr_emb_dict,cdr3s,longs,maxlen(cdr3s),append_onehot=p['append_oh'],device=device)

        # Second chain (if given)
        if p['two_chains']:
            cdr3s, longs,_,_ = get_seqdata(usetype, p,p['h_epitope'],p['h_cdr32'],p['h_long2'])
            if 'cdr3' == p['embedtype']: # Use CDR3 sequence without context
                embeddings2 = get_embeddings(tcr_emb_dict2,cdr3s,cdr3s,maxlen(cdr3s),append_onehot=p['append_oh'],device=device)
            elif 'tcr' == p['embedtype']: # Use full TCR sequence
                embeddings2 = get_embeddings(tcr_emb_dict2,longs,longs,maxlen(longs),append_onehot=p['append_oh'],device=device)
            else: # cdr3+context: Use CDR3 sequence with full context (recommended)
                embeddings2 = get_embeddings(tcr_emb_dict2,cdr3s,longs,maxlen(cdr3s),append_onehot=p['append_oh'],device=device)

            loader_u = DataLoader(tcrDatasetAB(embeddings,embeddings2,labels_ar),batch_size=p['batch_size'],shuffle=usetype=='train')
        else:
            loader_u = DataLoader(tcrDataset(embeddings,labels_ar),batch_size=p['batch_size'],shuffle=usetype=='train')

        loader[usetype] = loader_u
        n_categories[usetype] = labels_ar.shape[1]
        n_features = embeddings.shape[1]

    return loader, n_categories, n_features


##### METRICS ##################################################################
def get_aurocs_aps(labels,y_pred):
    """Get dictionaries of auroc and ap-scores with separate scores for each epitope label
        input: labels: """
    n_labels = labels.shape[1]

    roc_auc = dict()
    for i in range(n_labels):
        roc_auc[i] = roc_auc_score(labels[:, i], y_pred[:, i])

    roc_auc['micro'] = roc_auc_score(labels, y_pred, average='micro')
    roc_auc['macro'] = roc_auc_score(labels, y_pred, average='macro')


    average_precision = dict()
    for i in range(n_labels):
        average_precision[i] = average_precision_score(labels[:, i], y_pred[:, i])

    average_precision['micro'] = average_precision_score(labels, y_pred,average='micro')
    average_precision['macro'] = average_precision_score(labels, y_pred,average='macro')

    return roc_auc, average_precision

##### TRAINING #################################################################
def get_positive_weights(loader_train):
    """Compute weights for positive labels to account for class imbalance in training data """
    labels=[]
    for batch in loader_train:
        labels.append(batch['labels'])
    labels=torch.cat(labels,axis=0)
    pos_weight = (labels.shape[0]/torch.sum(labels,axis=0)) / labels.shape[1]
    return pos_weight

def iterate_model_batches_swa(model,loader,p,device):

    n_iters = p['iters_adam'] + p['iters_swa']
    swa_start = p['iters_adam']
    pos_weight = get_positive_weights(loader['train']) if p['use_pos_weight'] else None

    if p['binary'] is False:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else: # in case of binary classification, use different loss function
        criterion_0 = torch.nn.CrossEntropyLoss(weight=pos_weight)
        criterion = lambda output, labels: criterion_0(output,torch.argmax(labels,axis=1))

    # Keep track of losses for plotting
    current_loss = 0
    iter=1
    model.train()

    my_list = ['dense_i.weight', 'dense_i.bias']
    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
    optimizer = torch.optim.Adam([{'params': base_params},
        {'params': params, 'lr': p['lr_linear']}], lr=p['lr_conv'], betas=p['betas']) # 'lr': 0.01

    swa_model = AveragedModel(model)
    swa_model.to(device)

    swa_model.train()
    scheduler = CosineAnnealingLR(optimizer, T_max=p['T_anneal']) # T_max: max number of iterations
    swa_scheduler = SWALR(optimizer,anneal_strategy=p['anneal_strategy'], swa_lr=p['lr_swa'],anneal_epochs=p['T_anneal_swa'])

    loss_record, loss_record_test = [], []
    auroc_record, aupr_record = [], []
    while iter <= n_iters: # Can go over n_iters, if n_iters % (n_tcrs/n_batch_size) != 0
        for sample in loader['train']:

            model.train()
            optimizer.zero_grad()

            if p['two_chains']:
                output = model(sample['embeddings1'],sample['embeddings2'])
            else:
                output = model(sample['embeddings'])

            loss_train = criterion(output, sample['labels'])
            loss_train.backward()
            optimizer.step()

            if iter > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            loss= loss_train.item()
            current_loss += loss

            # Add current loss avg to list of losses
            if p['save_intermediate'] and iter % p['print_every'] == 0:

                loss_record.append(current_loss / (p['print_every']))
                current_loss = 0
                if p['lossfile_test']!='None':
                    model.eval()
                    loss,pred,nseqs,labels=[],[],[],[]
                    for sample_test in loader['test']:
                        if p['two_chains']:
                            output = model(sample_test['embeddings1'],sample_test['embeddings2'])
                        else:
                            output = model(sample_test['embeddings'])
                        loss.append(criterion(output,sample_test['labels']).item())
                        pred.append(np.squeeze(output.detach().cpu().numpy()))
                        labels.append(sample_test['labels'].clone().cpu().numpy())
                        nseqs.append(len(sample_test['labels']))

                    loss_record_test.append(np.sum([n*l for n,l in zip(nseqs,loss)])/np.sum(nseqs))
                    y_score=toprob(np.concatenate(pred),binary=p['binary'])
                    labels = np.concatenate(labels)
                    auroc_record.append(roc_auc_score(labels, y_score, average='macro'))
                    aupr_record.append(average_precision_score(labels, y_score,average='macro'))
            iter+=1

    if p['lossfile']!='None':
        with open(p['lossfile'],'a') as f:
            writer=csv.writer(f,delimiter='\t')
            writer.writerow(p['table'][1]+['Loss']+loss_record)
    if p['lossfile_test']!='None':
        with open(p['lossfile_test'],'a') as f:
            writer=csv.writer(f,delimiter='\t')
            writer.writerow(p['table'][1]+['Loss']+loss_record_test)
            writer.writerow(p['table'][1]+['AUROC']+auroc_record)
            writer.writerow(p['table'][1]+['AUPR']+aupr_record)

    #torch.optim.swa_utils.update_bn(loader['train'], swa_model)
    # We have a custom dataloader, so we'll do a forward pass instead to update statistics:
    swa_model.train()
    with torch.no_grad():
        for sample in loader['train']:
            ni = sample['embeddings1'].shape[0] if p['two_chains'] else sample['embeddings'].shape[0]
            for i in range(ni):
                if p['two_chains']:
                    _ = swa_model(sample['embeddings1'][i:i+1,:,:],sample['embeddings2'][i:i+1,:,:])
                else:
                    _ = swa_model(sample['embeddings'][i:i+1,:,:])

    return swa_model

##### LOAD MODEL ##############################################################
def construct_model(p,n_feat,n_labels,device):
    if p['two_chains']:
        model = models.CNN3AB(n_feat,n_labels,kernel_sizes=p['kernel_sizes'],
                dos=p['dropouts'],pool=p['pool'])
    else:
        model = models.CNN3(n_feat,n_labels,kernel_sizes=p['kernel_sizes'],
                dos=p['dropouts'],pool=p['pool'])

    model.to(device)

    return model

def load_model(model_file,p,n_feat,n_labels,device):
    model = construct_model(p,n_feat,n_labels,device)
    swa_model = AveragedModel(model)

    #model_file = p['model_folder']+'/statedict_'+p['name']+'_'+str(p['fold_num'])
    swa_model.load_state_dict(torch.load(model_file,map_location=device))
    swa_model.to(device)
    swa_model.eval();

    return swa_model


##### PREDICTIONS #############################################################
def get_yscore(model,loader,probs=True,binary=False,useAB=False):
    model.eval()
    yscores,labels = [],[]
    for sample in loader:
        if useAB:
            output = model(sample['embeddings1'],sample['embeddings2'])
        else:
            output = model(sample['embeddings'])
        yscores.append(output.detach().cpu().numpy())
        labels.append(sample['labels'].cpu().numpy())
    yscores=np.concatenate(yscores)
    if probs:
        yscores=toprob(yscores,binary=binary)

    return yscores,np.concatenate(labels)

##### PROCESS SEQUENCE DATA ####################################################
