# TCRconv
TCRconv is a deep learning model for predicting recognition between T cell receptors and epitopes. TCRconv is formulated as a multilabel predictor and uses protBERT embeddings for the TCRs and convolutional neural networks for the prediction.

![TCRconv pipeline](TCRconv-pipeline.jpeg)

## Usage
Do the following step to use TCRconv

### Optional: create virtual env
python3 -m venv tcrconv-env
#### Activate virtual env
source tcrconv-env/bin/activate (Linux, macOS) or tcrconv-env/Scripts/activate (Win)

### Install requirements
pip install -r requirements.txt

### Install tcrconv
pip install .

### to allow editing, instead use
pip install -e .


## Predictor



## LM: TCR-BERT

Collection of scripts to use the BERT model from  [ProtTrans](https://github.com/agemagician/ProtTrans)

Also contains the dataset preparation for further fine-tuning:
* Task fine-tuning: using a \<CLS\> token at the beginning of sequences, predict epitope specificity of TCR sequences from [VDJDB](https://vdjdb.cdr3.net/).
* Further fine-tuning: using a collection of data from [VDJDB](https://vdjdb.cdr3.net/), [Dash et al.](https://www.nature.com/articles/nature22383) and [Emerson et. al](https://www.nature.com/articles/ng.3822).

### Dummy Dataset

In this github repo, only a small collection of 2000 sequences from [VDJDB](https://vdjdb.cdr3.net/) is made available because of github storage constraints. Please download the resources from [VDJDB](https://vdjdb.cdr3.net/), [Dash et al.](https://www.nature.com/articles/nature22383) and [Emerson et. al](https://www.nature.com/articles/ng.3822) and refer to [TCR-preprocessing-script-fromgit]() for bert fine-tuning.


### Original and tuned models usage

Useful examples of how to load and use the BERT models depending on the application requirements are found in demo.py. The useful functions used in this are found in bert_mdl.py

### Further fine-tuning

Refer to fine_tune_bert.py script. the "--create_data" argument should be used the first time this model is ran. When "--tune_epitope_specificity", the epitope specificity annotated TCRs from [VDJDB](https://vdjdb.cdr3.net/), [Dash et al.](https://www.nature.com/articles/nature22383) and [Emerson et. al](https://www.nature.com/articles/ng.3822) are used to fine-tune the original BERT model. When this argument is not used, the model automatically furthe fine-tunes (trains the BERT model further on TCR-only sequences).

#### Dependencies

Python 3.5+ is required.
Please install python dependencies with "pip install -r requirements.txt"

### Acknowledgments

Inspiration, code snippets, etc.
* [ProtTrans](https://github.com/agemagician/ProtTrans)

Data:
* [VDJDB](https://vdjdb.cdr3.net/)
* [Dash et al.](https://www.nature.com/articles/nature22383) 
* [Emerson et. al](https://www.nature.com/articles/ng.3822).
