
import pickle

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim

import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel
from torchnlp.utils import collate_tensors

import pandas as pd
from test_tube import HyperOptArgumentParser
import os
import requests
from tqdm.auto import tqdm
from collections import OrderedDict
import logging as log
import numpy as np


class ProtBertClassifier(pl.LightningModule):
    """
    # https://github.com/minimalist-nlp/lightning-text-classification.git
    Sample model to show how to use BERT to classify sentences.
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams) -> None:
        super(ProtBertClassifier, self).__init__()
        self.hparams = hparams
        self.batch_size = 32

        self.modelFolderPath ='LM/ProtBert/'
        self.vocabFilePath = os.path.join(self.modelFolderPath, 'vocab.txt')

        self.extract_emb = False

        # build model
        self.__download_model()

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        self.freeze_encoder()

        if self.hparams.tune_epitope_specificity:
            self.classification_head = nn.Sequential(
                nn.Linear(self.encoder_features * 4, 22),
                nn.Tanh()
            )
        else:
            self.classification_head = nn.Sequential(
                nn.Linear(self.encoder_features, 25),
                nn.Tanh()
            )


    def __download_model(self) -> None:
        # Provided by the authors of Prottrans
        modelUrl = 'https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1'
        configUrl = 'https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1'
        vocabUrl = 'https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1'


        modelFolderPath = self.modelFolderPath
        modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
        configFilePath = os.path.join(modelFolderPath, 'config.json')
        vocabFilePath = self.vocabFilePath

        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)

        def download_file(url, filename):
            response = requests.get(url, stream=True)
            with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                               total=int(response.headers.get('content-length', 0)),
                               desc=filename) as fout:
                for chunk in response.iter_content(chunk_size=4096):
                    fout.write(chunk)

        if not os.path.exists(modelFilePath):
            download_file(modelUrl, modelFilePath)

        if not os.path.exists(configFilePath):
            download_file(configUrl, configFilePath)

        if not os.path.exists(vocabFilePath):
            download_file(vocabUrl, vocabFilePath)

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        self.ProtBertBFD = BertModel.from_pretrained(self.modelFolderPath)
        self.encoder_features = 1024

        # Tokenizer
        self.tokenizer = BertTokenizer(self.vocabFilePath, do_lower_case=False)


    def __build_loss(self):
        """ Initializes the loss function/s. """
        self._loss = nn.CrossEntropyLoss()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for name, param in self.ProtBertBFD.named_parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.ProtBertBFD.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze_classification_head(self):
        for param in self.classification_head.parameters():
            param.requires_grad = True

    def predict(self, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.
        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.prepare_sample([sample], prepare_target=False)
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()
            predicted_labels = [
                self.label_encoder.index_to_token[prediction]
                for prediction in np.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]

        return sample

    # https://github.com/UKPLab/sentence-transformers/blob/eb39d0199508149b9d32c1677ee9953a84757ae4/sentence_transformers/models/Pooling.py
    def pool_strategy(self, features,
                      pool_cls=True, pool_max=True, pool_mean=True,
                      pool_mean_sqrt=True):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if pool_cls:
            output_vectors.append(cls_token)
        if pool_max:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        return output_vector

    def extract_embeddnings(self, sample):
        inputs = self.tokenizer.batch_encode_plus(sample["seq"],
                                                  add_special_tokens=True,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.hparams.max_length)
        return self.forward(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'],
                            return_embeddings=True).cpu().numpy()

    def forward(self, input_ids, token_type_ids, attention_mask, target_positions=None, return_embeddings=False, cdr3b_sequences=None, long_sequences=None,
                cdr3a_sequences=None,long_sequences_a=None):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        input_ids = torch.tensor(input_ids, device=self.device)
        batch_size, seq_dim = input_ids.shape[0], input_ids.shape[1]
        attention_mask = torch.tensor(attention_mask, device=self.device)
        word_embeddings = self.ProtBertBFD(input_ids,
                                           attention_mask)[0]
        if cdr3b_sequences is not None:
            seq_embs = []
            # extract only the cdr3b embeddings
            for seq_num in range(len(cdr3b_sequences)):
                start_cdr3 = long_sequences[seq_num].find(cdr3b_sequences[seq_num])
                end_cdr3 = start_cdr3 + len(cdr3b_sequences[seq_num])
                seq_embs.append(word_embeddings[seq_num][start_cdr3:end_cdr3])
            # First half corresponds to cdr3b/longb seeqs, second to cdr3a/longa
            for seq_num in range(len(cdr3a_sequences)):
                start_cdr3 = long_sequences_a[seq_num].find(cdr3a_sequences[seq_num])
                end_cdr3 = start_cdr3 + len(cdr3a_sequences[seq_num])
                seq_embs.append(word_embeddings[seq_num+ len(cdr3b_sequences)][start_cdr3:end_cdr3])
            # padd them with zeros
            # TODO check if it pads at the end
            embs = torch.nn.utils.rnn.pad_sequence(seq_embs, batch_first=True)
            # throw the embeddings in the classifier and get predictions
            if cdr3a_sequences:
                cdr3a_start = len(embs)//2
                cdr3b = embs[:cdr3a_start]
                cdr3b = cdr3b.permute(0,2,1)
                cdr3a = embs[cdr3a_start:]
                cdr3a  = cdr3a.permute(0,2,1)
                preds = self.classification_head(cdr3b, cdr3a)
            else:
                preds = self.classification_head(embs.permute(0, 2, 1))
            return preds

        if self.extract_emb:
            # used for extracting the actual embeddings after tuning
            return word_embeddings

        if self.hparams.tune_epitope_specificity:
            # we dont want any pooling (only extract the embeddings corresponding to the masked inputs)
            pooling = self.pool_strategy({"token_embeddings": word_embeddings,
                                          "cls_token_embeddings": word_embeddings[:, 0],
                                          "attention_mask": attention_mask,
                                          })
            return self.classification_head(pooling)
        word_embeddings = word_embeddings.reshape(-1, 1024)
        seq_delim = torch.tensor(list(range(batch_size)), device=self.device) * seq_dim
        seq_delim = seq_delim.reshape(-1, 1)
        target_positions = torch.tensor(target_positions, device=self.device).reshape(-1, len(target_positions[0]))
        target_positions = target_positions + seq_delim
        target_positions = target_positions.reshape(-1)
        prediction_embeddings = word_embeddings[target_positions]
        out = self.classification_head(prediction_embeddings)
        return out

    def loss(self, predictions: torch.tensor, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions, torch.tensor(targets["labels"], device=predictions.device))

    def encode_labels(self, labels):
        if self.hparams.tune_epitope_specificity:
            return labels
        vocab = {"L": 0, "A": 1, "G": 2, "V": 3, "E": 4, "S": 5,
                 "I": 6, "K": 7, "R": 8, "D": 9, "T": 10, "P": 11, "N": 12, "Q": 13, "F": 14, "Y": 15,
                 "M": 16, "H": 17, "C": 18, "W": 19, "X": 20, "U": 21, "B": 22, "Z": 23, "O": 24}
        bs = len(labels[0])
        all_labels = []
        for i in range(bs):
            all_labels.append([vocab[l[i]] for l in labels])
        return all_labels

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        inputs = self.tokenizer.batch_encode_plus(sample["seq"],
                                                  add_special_tokens=self.hparams.special_tokens,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.hparams.max_length)

        if not prepare_target:
            return inputs, {}

        # Prepare target:

        try:
            targets = {"labels": self.encode_labels(sample["target"])}
            if self.hparams.tune_epitope_specificity:
                return inputs, targets
            return inputs, targets, sample["target_pos"]
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        # inputs, targets = batch
        self.classification_head.train()
        self.ProtBertBFD.train()
        if self.hparams.tune_epitope_specificity:
            inputs, targets = batch
            model_out = self.forward(**inputs)
            loss_val = self.loss(model_out, targets)
            tqdm_dict = {"train_loss": loss_val}
            output = OrderedDict(
                {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})
            # can also return just a scalar instead of a dict (return loss_val)
            return output

        inputs, targets, target_positions = batch
        all_targets = []
        for tl in targets["labels"]:
            all_targets.extend(tl)
        targets["labels"] = all_targets

        bs = len(target_positions[0])
        all_target_pos = []
        for i in range(bs):
            all_target_pos.append([tp_[i] for tp_ in target_positions])
        target_positions = all_target_pos
        inputs["target_positions"] = target_positions

        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        if self.hparams.tune_epitope_specificity:
            inputs, targets = batch
            model_out = self.forward(**inputs)
            loss_val = self.loss(model_out, targets)
            y = targets["labels"]
            # y_hat = model_out["logits"]
            y_hat = model_out
            labels_hat = torch.argmax(y_hat, dim=1)
            # labels_hat = y_hat
            val_acc = self.metric_acc(labels_hat, y)

            output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc, })
            return output

        inputs, targets, target_positions = batch
        all_targets = []
        for tl in targets["labels"]:
            all_targets.extend(tl)
        targets["labels"] = all_targets
        bs = len(target_positions[0])
        all_target_pos = []
        for i in range(bs):
            all_target_pos.append([tp_[i] for tp_ in target_positions])
        target_positions = all_target_pos
        inputs["target_positions"] = target_positions
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)
        y = targets["labels"]
        y_hat = model_out
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = self.metric_acc(labels_hat, y)

        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc, })
        return output

    def validation_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """

        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()

        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }
        return result

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_test = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out

        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = self.metric_acc(labels_hat, y)

        output = OrderedDict({"test_loss": loss_test, "test_acc": test_acc, })
        return output

    def test_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc_mean = torch.stack([x['test_acc'] for x in outputs]).mean()

        tqdm_dict = {"test_loss": test_loss_mean, "test_acc": test_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "test_loss": test_loss_mean,
        }
        return result

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.ProtBertBFD.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

def parse_arguments():
    parser = HyperOptArgumentParser(
        strategy="random_search",
        description="Minimalist ProtBERT Classifier",
        add_help=True,
    )
    # the only arguments required by the model to function:
    #       max_length: should be set to be > maximum sequence length of the dataset.
    #       special_tokens: appends tokens like "<beggining_of_sequence>" and "<classification_token>" to the
    #                       sequence. When computing embeddings (not tuning the model) should be set to false.
    #       tune_epitope_specificity: when loading a model from a checkpoint, set this to be the same as the
    #                                 loaded checkpoint's training (further fine-tuning/ep specificity tuning)
    params = {'max_length': 1536, 'special_tokens': False, 'tune_epitope_specificity':False}
    hparams = parser.parse_known_args([])[0]
    hparams.__dict__.update(params)
    return hparams

def retrieve_model():
    """
        Function that parses arguments and return a model with those
        Outputs:
            ProtBertClassifier model object (the original model will be automatically downloaded when initialized under
            "model" directory
    """
    hparams = parse_arguments()
    # hparams = parse_known_args()
    model = ProtBertClassifier(hparams)
    return model


def extract_and_save_embeddings(model=None, data_f_n="vdj_human_unique_longs.csv", separator=",", sequence_col="long",
                                cdr3_col=None, emb_name="default_name", add_long_aa=-1, seqs_per_file=45000):
    """
        Useful for extracting embeddings of a set of sequences and saving them. This will speed up e.g. training with
        BERT embeddings as inputs (as the sequences will have their respective BERT embeddings computed only once)
        Parameters:
            model: loaded BERT model
            data_f_n: full path to a csv/tsv file containing the requested sequences
            separator: "," for csv and "\t" for tsv
            sequence_col: column name in the data_f_n file containing the sequences (as strings)
            cdr3_col (optional): name of the cdr3 column in the data_f_n file. When using only the cdr3 regions from the
                                 sequences, but computing the embeddings for them using context residues - using the
                                 sequences provided in sequence_col(other residues encoded by V, D and J genes)
            emb_name: name of the file that will contain the resulted embeddings. The names additionally have
                      "_id.bin" at the end of emb_name.
            add_long_aa: only use that amount of residues to the left and right of the cdr3 (from the sequence_col sequence)
                         when computing the embeddings
            seqs_per_file: save max seqs_per_file embeddings to one file (default 45000).
    """
    model = retrieve_model() if model is None else model
    model.cuda()
    model.to(torch.float32)
    data = pd.read_csv(data_f_n, sep=separator)
    if cdr3_col is None:
        long = data[sequence_col]
        cdr3b = long
    else:
        long, cdr3b = data[sequence_col].values, data[cdr3_col].values
    l2cdr3 = {ld:cdr3 for ld, cdr3 in zip(long, cdr3b)}
    cropped_seqs, sequences = [], []
    for t, l in zip(cdr3b, long):
        if add_long_aa != -1:
            cdr_start = l.find(t)
            cdr_end = cdr_start + len(t)
            cropeed_seq = l[cdr_start - add_long_aa: cdr_end + add_long_aa]
            cropped_seqs.append(' '.join(list(cropeed_seq)))
        else:
            cropped_seqs.append(' '.join(list(l)))
        sequences.append(l)
    inputs = model.tokenizer.batch_encode_plus(cropped_seqs[:10],
                                               add_special_tokens=False,
                                               padding=True,
                                               truncation=True,
                                               max_length=model.hparams.max_length)
    print("Beginning extraction of {} sequence embeddings. Embeddings will be saved to {}".format(len(sequences), emb_name))
    if add_long_aa == -1:
        print("All context will be considered")
    elif add_long_aa:
        print("number of aa for left and right will be taken for context: {}".format(add_long_aa))

    vdjdb_embeddings = {}
    count = 0
    model.extract_emb = True
    model.eval()
    file_index = 0
    for i in range(int(np.ceil(len(sequences)/ 100))):
        current_inputs = {}
        seqs = sequences[i * 100:(i + 1) * 100]
        crped_seqs = cropped_seqs[i * 100:(i+1) * 100]
        current_inputs['input_ids'] = inputs['input_ids'][i * 100:(i + 1) * 100]
        current_inputs['token_type_ids'] = inputs['token_type_ids'][i * 100:(i + 1) * 100]
        current_inputs['attention_mask'] = inputs['attention_mask'][i * 100:(i + 1) * 100]
        embedding = model.forward(**current_inputs)
        pickle.dump([seqs[:10], embedding[:10]], open("check_embs.bin", "wb"))
        exit(1)
        embedding = embedding.cpu().numpy()
        print("{} sequences have been extracted".format(i * 100))
        for seq_num in range(len(embedding)):
            seq_len = len(seqs[seq_num].replace(" ", ""))
            start_Idx = 1 if model.hparams.special_tokens else 0
            end_Idx = seq_len + 1 if model.hparams.special_tokens else seq_len
            seq_emd = embedding[seq_num][start_Idx:end_Idx]
            ld = seqs[seq_num]
            ld_cropped = crped_seqs[seq_num].replace(" ", "")
            cdr3 = l2cdr3[ld]
            start_cdr3 = ld_cropped.find(cdr3)
            end_cdr3 = start_cdr3 + len(cdr3)
            seq_emd = seq_emd[start_cdr3:end_cdr3]
            vdjdb_embeddings[seqs[seq_num]] = seq_emd.transpose(1,0)
        if i * 100 % seqs_per_file == 0 and i != 0:
            pickle.dump(vdjdb_embeddings, open(emb_name + "_{}.bin".format(file_index), "wb"))
            file_index += 1
            vdjdb_embeddings = {}
    if vdjdb_embeddings:
        pickle.dump(vdjdb_embeddings, open(emb_name + "_{}.bin".format(file_index), "wb"))

def get_saliency_for_batch(model, batch1, batch2, epitope2lbl=None):
    retain_grads = []
    long1_i, epitopes_i, cdr31_i = batch1
    long2_i, cdr32_i = batch2
    def hook_(self, grad_inp, grad_out):
        retain_grads.append((grad_out[0].cpu()))

    model.requires_grad = True
    model.unfreeze_encoder()

    # this takes the gradients wrt. (input_embedding(aa_seq) + pos_enc(aa_seq))
    handle = model.ProtBertBFD.embeddings.register_backward_hook(hook_)
    # handle = model.ProtBertBFD.encoder.register_backward_hook(hook_)

    current_inputs = {}

    split_long_bs = []
    for l in long1_i:
        split_long_bs.append(' '.join(list(l)))
    for l in long2_i:
        split_long_bs.append(' '.join(list(l)))

    inputs = model.tokenizer.batch_encode_plus(split_long_bs,
                                               add_special_tokens=False,
                                               padding=True,
                                               truncation=True,
                                               max_length=model.hparams.max_length)
    current_inputs['input_ids'] = inputs['input_ids']
    current_inputs['token_type_ids'] = inputs['token_type_ids']
    current_inputs['attention_mask'] = inputs['attention_mask']
    current_inputs['cdr3b_sequences'] = cdr31_i
    current_inputs['long_sequences'] = long1_i
    current_inputs['cdr3a_sequences'] = cdr32_i
    current_inputs['long_sequences_a'] = long2_i

    preds = torch.sigmoid(model.forward(**current_inputs))
    for seq_ind, p in enumerate(torch.argmax(preds,dim=1)):
        if epitope2lbl is not None:
            label_ind = epitope2lbl[epitopes_i[seq_ind]]
        preds[seq_ind, label_ind].backward(retain_graph=True)
    handle.remove()
    return retain_grads, torch.argmax(preds,dim=1).detach().cpu().numpy()


def duplicate_cross_reactive_tcrs(long1, epitopes, cdr31, long2=[], cdr32=[]):
    long1_, epitopes_, cdr31_, long2_, cdr32_ = [],[],[],[],[]
    if len(long2)>1:
        for l1, ep, c1, l2, c2 in zip(long1, epitopes, cdr31, long2, cdr32):
            for ep_ in ep.split(" "):
                long1_.append(l1)
                epitopes_.append(ep_)
                cdr31_.append(c1)
                long2_.append(l2)
                cdr32_.append(c2)
    else:
        for l1, ep, c1 in zip(long1, epitopes, cdr31):
            for ep_ in ep.split(" "):
                long1_.append(l1)
                epitopes_.append(ep_)
                cdr31_.append(c1)

    return long1_, epitopes_, cdr31_, long2_, cdr32_

def get_saliency_map(model, p, epitope2lbl=None,abs=True):
    # model.ProtBertBFD.requires_grad=True
    # onelasttime
    batch_size = p['batch_size']
    chains = p['chains']


    model = retrieve_model() if model is None else model
    model.zero_grad()
    model.ProtBertBFD.zero_grad()
    model.ProtBertBFD.eval()
    model.extract_emb = True
    model.eval()

    data = pd.read_csv(p['dataset'], sep=p['delimiter'])
    epitopes = data[p['h_epitope']].values

    long1 = data[p['h_long1']].values
    cdr31 = data[p['h_cdr31']].values

    if len(chains)>1:
        long2 = data[p['h_long2']].values
        cdr32 = data[p['h_cdr32']].values

    if epitope2lbl is not None:
        # if max pred is required, gradients are of the max-pred, otherwise multiple sets of gradients wrt to each
        # associated epitope for a TCR are required
        if len(chains)>1:
            long1, epitopes, cdr31, long2, cdr32 = duplicate_cross_reactive_tcrs(long1, epitopes, cdr31, long2, cdr32)
        else:
            long1, epitopes, cdr31, long2, cdr32 = duplicate_cross_reactive_tcrs(long1, epitopes, cdr31)

    saliency_results = []
    n_seqs = len(long1)
    n_batches = int(np.ceil(n_seqs/ batch_size))
    for i in range(n_batches):
        print('batch ',i,'of',n_batches)
        i_start = i * batch_size
        i_end = min((i + 1) * batch_size, n_seqs)
        long1_i = long1[i_start : i_end]
        epitopes_i = epitopes[i_start : i_end]
        cdr31_i = cdr31[i_start : i_end]
        if len(chains)>1:
            long2_i = long2[i_start : i_end]
            cdr32_i = cdr32[i_start : i_end]
        else:
            long2_i = []
            cdr32_i = []
        input_grads, preds = get_saliency_for_batch(model, [long1_i,epitopes_i,cdr31_i], [long2_i, cdr32_i], epitope2lbl)
        model.zero_grad()

        # the gradients are of pred_i for element i in batch wrt. all inputs of size (batch_size, seq_len, emb_dim),
        # meaning all off-diagonal grad [y_i(batch_element_j,:,:)] i!=j are simply 0;
        # extract the (useful) diagonal elements grad [y_i(batch_element_i,:,:)]
        input_grads_cdr31 = [input_grads[i][i,:,:].detach().cpu().numpy() for i in range(len(input_grads))]
        if abs:
            input_grads_cdr31 = [np.sum(np.abs(ig), axis=1) for ig in input_grads_cdr31]
        else:
            input_grads_cdr31 = [np.sum(ig, axis=1) for ig in input_grads_cdr31]

        if len(chains)>1:
            # input tensors to ProtBERT are [long1_1,long1_2, ..., long1_n, long2_1, long2_2, ..., long2_n] so gradiets
            # wrt. cdr32 come on the second half of the tensors: i+len(cdr31)

            input_grads_cdr32 = [input_grads[i][i+len(input_grads_cdr31),:,:].detach().cpu().numpy() for i in range(len(input_grads))]
            if abs:
                input_grads_cdr32 = [np.sum(np.abs(ig),axis=1) for ig in input_grads_cdr32]
            else:
                input_grads_cdr32 = [np.sum(ig,axis=1) for ig in input_grads_cdr32]
        else:
            input_grads_cdr32 = [None for _ in range(len(input_grads_cdr31))]


        # gather all information
        if len(chains)>1:
            for ep, pred, ig_cdr31, ig_cdr32, l1, c1, l2, c2 in zip(epitopes_i, preds, input_grads_cdr31, input_grads_cdr32, long1_i, cdr31_i, long2_i, cdr32_i):
                saliency_results.append([ep, pred, ig_cdr31, ig_cdr32, l1, c1, l2, c2])
        else:
            for ep, pred, ig_cdr31, l1, c1 in zip(epitopes_i, preds, input_grads_cdr31, long1_i, cdr31_i):
                saliency_results.append([ep, pred, ig_cdr31, l1, c1])
        # TODO remove at some point. barplots of saliency maps, useful for debugging
        # process them by [e_l: el = abs( sum_{i=1,emb_dim}; l=1,...,seq_len]
        # input_grads = [np.sum(np.abs(ie), axis=1) for ie in input_grads]
        # for i in range(batch_size):
        #     cdr, lng = cdr3b_b[i], long_b[i]
        #     cdr3_start,cdr3_end = lng.find(cdr), lng.find(cdr)+len(cdr)
        #     plt.bar(list(range(cdr3_start)), input_grads_cdr3b[i][:cdr3_start], color='blue')
        #     plt.bar(list(range(cdr3_start, cdr3_end)), input_grads_cdr3b[i][cdr3_start:cdr3_end], color='orange')
        #     plt.bar(list(range(cdr3_end, len(lng))), input_grads_cdr3b[i][cdr3_end:len(lng)], color='blue')
        #     plt.title(lng[0])
        #     plt.show()
        #
        #     if cdr3a:
        #         cdra, lnga = cdr3a[i], longa[i]
        #         cdr3_start,cdr3_end = lnga.find(cdra), lnga.find(cdra)+len(cdra)
        #         plt.bar(list(range(cdr3_start)), input_grads_cdr3a[i][:cdr3_start], color='blue')
        #         plt.bar(list(range(cdr3_start, cdr3_end)), input_grads_cdr3a[i][cdr3_start:cdr3_end], color='orange')
        #         plt.bar(list(range(cdr3_end, len(lng))), input_grads_cdr3a[i][cdr3_end:len(lng)], color='blue')
        #         plt.title(lng[0])
        #         plt.show()
        # print(input_grads[0].shape)
        # exit(1)

    return saliency_results



def compute_embs(model, seqs, cdr3s=None):
    """
        This function is useful when computing the bert embeddings in an "online" manner. E.g. if there is a lot of
        sequence data (millions), saving the BERT embeddings for each of those would require a lot of storage space.
        In that case, use compute_embs for each sequence batch during training
        Parameters:
            model: loaded BERT model
            seqs: sequences that need to be extracted
            cdr3s: None or list of cdr3 sequences contained in the longer seqs. If a list is given, the model
                computes the embeddings from seqs and accordingly extracts the cdr3 sequences
        Outputs:
            embedding: list of numpy arrays of (1024 x L) dimension for each input sequence
    """
    model.extract_emb = True
    def remove_special_tkns(embs, use_special_tkns=False):
        if not use_special_tkns:
            return  embs
        else:
            return [e[1:-1] for e in embs]

    def extract_cdr3(embs, cdr3s, seqs):
        if cdr3s is None:
            return embs
        else:
            cdr3_embs = []
            for e,c,s in zip(embs, cdr3s, seqs):
                istart = s.find(c)
                cdr3_embs.append(e[istart:istart+len(c)])
            return cdr3_embs


    cropped_seqs = []
    for s in seqs:
        cropped_seqs.append(' '.join(list(s)))
    import time
    inputs = model.tokenizer.batch_encode_plus(cropped_seqs,
                                               add_special_tokens=model.hparams.special_tokens,
                                               padding=True,
                                               truncation=True,
                                               max_length=model.hparams.max_length)
    embedding = model.forward(**inputs)
    embedding = embedding.cpu().numpy()
    embedding = remove_special_tkns(embedding, model.hparams.special_tokens)
    embedding = extract_cdr3(embedding, cdr3s, seqs)
    embedding = [e.transpose(1,0) for e in embedding]
    return embedding
