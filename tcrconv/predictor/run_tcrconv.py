
import torch
import numpy as np

from os import path, mkdir
import utils

# Read arguments
p = utils.get_parameter_dict()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mode =p['mode'].lower()
# Get dataloaders
if mode=='cv':
    usetypes=['train','test']
elif mode=='train':
    usetypes=['train']

loader, n_categories, n_feat = utils.get_dataloaders(p,device=device,usetypes=usetypes)
# Create model
model = utils.construct_model(p,n_feat,n_categories['train'],device)
# Create requested loss and result files if they don't exist yet
utils.create_resultfiles(p,n_categories)
# Model training
model = utils.iterate_model_batches_swa(model,loader,p,device)
# Save model if model_folder is given
if p['model_folder'] != 'None':
    if not path.isdir(p['model_folder']):
        mkdir(p['model_folder'])
    modelfile = 'statedict_'+p['name'] + ('_'+str(p['fold_num']))*(p['mode']=='cv') + '.pt'
    torch.save(model.state_dict(), p['model_folder']+'/'+modelfile)

# Make predictions and save results
if p['resultfile'] != 'None':
    y_score, labels = utils.get_yscore(model,loader['test'],useAB=p['two_chains'])
    y_score=y_score[:,:n_categories['test']]
    utils.save_results(y_score,labels,p,n_categories['test'])
