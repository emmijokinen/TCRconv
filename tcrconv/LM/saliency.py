import pickle
import os
from bert_mdl import retrieve_model, get_saliency_map
from tcrconv.predictor.utils import get_parameter_dict, load_model
import torch
from numpy import load as np_load


if __name__=="__main__":

    # # load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = get_parameter_dict(predict=True)

    # Get epitope information
    epis_u = np_load(p['epitope_labels'])
    epitope2lbl = {e:i for i,e in enumerate(epis_u)}

    # Get TCRconv model
    n_labels = len(epis_u)
    classification_head = load_model(p['model_file'], p, p['num_features'], n_labels, device)

    # Get BERT model and TCRconv as classification head
    model = retrieve_model()
    model.classification_head = classification_head
    model.to(device)

    # Compute and save saliency
    saliency_results = get_saliency_map(model, p, epitope2lbl=epitope2lbl,abs=False)
    pickle.dump(saliency_results, open(p['predfile'], "wb"))
