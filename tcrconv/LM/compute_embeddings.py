from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter,ArgumentTypeError
from bert_mdl import retrieve_model, extract_and_save_embeddings
import torch


def get_parameter_dict():
    parser = ArgumentParser(description='TCRconv parameters',formatter_class=ArgumentDefaultsHelpFormatter)

    # Data
    parser.add_argument('--name', type=str,
                        help='name for the model. Will be used if the model or results are saved.')
    parser.add_argument('--dataset', type=str,
                        help='filename of the used dataset')
    parser.add_argument('--delimiter',type=str, default=',',
                        help='Column delimiter in dataset file.')

    # First chain
    parser.add_argument('--h_cdr3',type=str, default='CDR3B',
                        help='Column name for CDR3 of chain 1 in dataset file')
    parser.add_argument('--h_long',type=str, default='LongB',
                        help='Column name for Long TCR-sequence of chain 1 in dataset file')

    # Training parameters: general
    parser.add_argument('--seqs_per_file',type=int, default=50000,
                        help='Maximum number of sequences in one embedding file. If there are more sequences, \
                        the embeddings will be split into several files.')


    p = parser.parse_args()
    #p = p[0].__dict__ # Let's use this as a dictionary
    p = p.__dict__ # Let's use this as a dictionary

    return p


p = get_parameter_dict()

# # load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = retrieve_model().to(device)

# extract and save some embeddings
extract_and_save_embeddings(model, data_f_n=p['dataset'], sequence_col=p['h_long'], cdr3_col=p['h_cdr3'], seqs_per_file=p['seqs_per_file'], emb_name=p['name'],separator=p['delimiter'])
print("finished")
