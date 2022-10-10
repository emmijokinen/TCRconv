# TCRconv
TCRconv is a deep learning model for predicting recognition between T cell receptors and epitopes. TCRconv is formulated as a multilabel predictor and uses protBERT embeddings for the TCRs and convolutional neural networks for the prediction. 
It computes embeddings for the full TCR sequence (determined by the rearranged V(D)J genes) and extracts the embedding corresponding to the CDR3-sequence. This CDR3 embedding is transfused with information from its context, TCR regions outside the CDR3. This provides at least as accurate predictions as using the embeddings for the full TCR but requires less computational resources. TCRconv's predictor part uses convolutions to extract contextualized motifs and provides state-of-the-art TCR-epitope prediction accuracy. For a detailed description, see our paper *Predicting recognition between T cell receptors and epitopes using contextualized motifs* \[1\]

![TCRconv pipeline](TCRconv-pipeline.jpeg)

## Installation
Do the following step to use TCRconv. Python 3.5+ is required.

**Optional: create virtual env and activate it** \
python3 -m venv tcrconv-env \
source tcrconv-env/bin/activate (Linux, macOS) or tcrconv-env/Scripts/activate (Win)

**Install requirements** \
pip install -r requirements.txt

**Install tcrconv** \
pip install .

**to allow editing, instead use** \
pip install -e .

## Usage
(After preprocessing the data) We recommend to use gpus with TCRconv, especially when computing embeddings with protBERT.
### Preprocessing data
* We have used data downloaded from from VDJdb \[2\]
* preprocess-data.ipynb shows how this type of data can be utilized and processed for TCRconv. 
  * For example, it's shown how datasets used in \[1\] were created.
  * How to get stratified cross-validation folds
  * Also some visualizations on the data are provided for visual inspection
### Creating embeddings
* After a dataset is created, LM/compute_embeddings.py can be used for computing an embedding dictionary.
  * On commandline, run with --help to see descriptions for the possible inputs
  * For an example, see scripts/run_compute_embs-b-small.sh
### Training a TCRconv model
* When a dataset and corresponding embeddings are created, a model can be trained with predictor/train_tcrconv.py
  * A single model can be trained with mode 'train'
  * Cross-validation can be used with mode 'cv'
  * On commandline, run with --help to see descriptions for the possible inputs
  * For an example, see scripts/cv-b-small.sh
### Predictions with TCRconv
* When a TCRconv-model has been trained, it can be used to predict if new TCRs recognize the epitopes the model was trained for
  * Predictions can be made with a precomputed embedding dictionary (This is probably better if you plan to use the same embeddings multiple times)
  * It is also possible to compute the embeddings on the go (This is good if you want to use the embeddings only once or have massive amounts of data)
  * On commandline, run with --help to see descriptions for the possible inputs
  * For an example, see scripts/pred-b-small.sh

## Pretrained models
Torch state dictionaries for models trained with the VDJdb-b-small, VDJdb-b-large, and VDJdb-ab-large are available at models-folder. They are trained with the default parameters, according to shell-scripts in scripts folder (train-b-small, train-b-large, train-ab-large).


## References
\[1\] Jokinen, E., Dumitrescu, A., Huuhtanen, J., Heinonen M., Gligorijevic, V., Bonneau, R., Mustjoki, S., Lähdesmäki, H. Determining recognition between T cell receptors and epitopes using contextualized motifs, *submitted* (2022). Preprint available at bioRxiv (https://doi.org/10.1101/2022.05.23.493034) \
\[2\] Bagaev, Dmitry V., et al. VDJdb in 2019: database extension, new analysis infrastructure and a T-cell receptor motif compendium. *Nucleic Acids Research* **48.D1** (2020): D1057-D1062. https://vdjdb.cdr3.net 

## Predictor
Codes for the multilabel predictor.


## LM: protBERT for TCR sequences
Codes for the Language model.

Collection of scripts to use the protBERT model from  [ProtTrans](https://github.com/agemagician/ProtTrans)

Also contains the dataset preparation for further fine-tuning:
* Task fine-tuning: using a \<CLS\> token at the beginning of sequences, predict epitope specificity of TCR sequences from [VDJDB](https://vdjdb.cdr3.net/).
* Further fine-tuning: using a collection of data from [VDJDB](https://vdjdb.cdr3.net/), [Dash et al.](https://www.nature.com/articles/nature22383) and [Emerson et. al](https://www.nature.com/articles/ng.3822).

### Dummy Dataset

In this github repo, only a small collection of 2000 sequences from [VDJDB](https://vdjdb.cdr3.net/) is made available because of github storage constraints. Please download the resources from [VDJDB](https://vdjdb.cdr3.net/), [Dash et al.](https://www.nature.com/articles/nature22383) and [Emerson et. al](https://www.nature.com/articles/ng.3822) and refer to [TCR-preprocessing-script-fromgit]() for bert fine-tuning.

### Original and tuned models usage

Useful examples of how to load and use the BERT models depending on the application requirements are found in demo.py. The useful functions used in this are found in bert_mdl.py

### Further fine-tuning

Refer to fine_tune_bert.py script. the "--create_data" argument should be used the first time this model is ran. When "--tune_epitope_specificity", the epitope specificity annotated TCRs from [VDJDB](https://vdjdb.cdr3.net/), [Dash et al.](https://www.nature.com/articles/nature22383) and [Emerson et. al](https://www.nature.com/articles/ng.3822) are used to fine-tune the original BERT model. When this argument is not used, the model automatically further fine-tunes (trains the BERT model further on TCR-only sequences).

### Acknowledgments

Inspiration, code snippets, etc.
* [ProtTrans](https://github.com/agemagician/ProtTrans)

Data:
* [VDJDB](https://vdjdb.cdr3.net/)
* [Dash et al.](https://www.nature.com/articles/nature22383) 
* [Emerson et. al](https://www.nature.com/articles/ng.3822).

## ScRNA+TCRab-seq analysis of bronchoalveolar lavage samples
Implementation for the scRNA+TCRab-seq data analysis is available at https://github.com/janihuuh/tcrconv_manu 
