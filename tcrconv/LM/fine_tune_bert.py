
import pickle
import random
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer

from transformers import BertTokenizer, BertModel

from torchnlp.encoders import LabelEncoder
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import collate_tensors

import pandas as pd
from test_tube import HyperOptArgumentParser
import os
import requests
from tqdm.auto import tqdm
from datetime import datetime
from collections import OrderedDict
import logging as log
import numpy as np


def gpu_acc_metric(y_hat, labels):
    # A torch way of extracting accuracy. Used like this for gpu compatibility
    with torch.no_grad():
        acc = sum(y_hat == torch.tensor(labels, device=y_hat.device)).to(dtype=torch.float) / y_hat.shape[0]
    return acc

def create_epitope_tuning_files(relative_data_path):
    """
        Creates input sequences and epitope specificity label dataset for task fine-tuning and saves the files in the
        data folder.
            Inputs:
                relative_data_path: path from the main script to the data folder
    """
    def split_data(eps):
        ep2id = {}
        for ind, e in enumerate(eps):
            if e not in ep2id:
                ep2id[e] = [ind]
            else:
                ep2id[e].append(ind)
        all_test_inds, all_val_inds, all_train_inds = [], [], []
        for _, inds in ep2id.items():
            test_inds_ = random.sample(inds, max(2, round(len(inds) * 0.05)))
            val_inds_ = test_inds_
            train_inds_ = [i for i in inds if i not in test_inds_]
            all_test_inds.extend(test_inds_)
            all_train_inds.extend(train_inds_)
            all_val_inds.extend(val_inds_)
        return all_train_inds, all_val_inds, all_test_inds

    epitope_dataset = pd.read_csv(relative_data_path+'data/vdj_human_uniques_long.csv')
    epitope, sequences = epitope_dataset['epitope'].values, epitope_dataset['long'].values
    all_train_inds, all_val_inds, all_test_inds = split_data(epitope)
    train_seq, train_ep, test_seq, test_ep = sequences[all_train_inds], epitope[all_train_inds], \
                                             sequences[all_test_inds], epitope[all_test_inds]

    train_df = pd.DataFrame(train_seq, train_ep)
    test_df = pd.DataFrame(test_seq, test_ep)
    valid_df = pd.DataFrame(test_seq, test_ep)

    train_df.to_csv(relative_data_path + "data/epitope_seq_train.csv")
    test_df.to_csv(relative_data_path + "data/epitope_seq_test.csv")
    valid_df.to_csv(relative_data_path + "data/epitope_seq_valid.csv")


def create_bert_further_tuning_files(relative_data_path="."):
    """
        Extracts raw sequences for further tuning the BERT model on TCR-only data. The 15% masking is done
        in the dataset class, later
    """
    data = pd.read_csv(os.path.join(relative_data_path, "data/vdj_human_uniques_long.csv"))
    seqs = data['long'].values
    train_format_seqs = []
    for s in seqs:
        train_format_seqs.append(" ".join([s_ for s_ in s]))
    test_inds = random.sample(list(range(len(train_format_seqs))), int(0.2 * len(train_format_seqs)))
    valid_seqs, test_seqs = test_inds[:len(test_inds)//2], test_inds[len(test_inds)//2:]
    train_seqs = list(set(list(range(len(train_format_seqs)))) - set(test_inds))
    valid_data, test_data, train_data = [train_format_seqs[i] for i in valid_seqs], \
                                        [train_format_seqs[i] for i in test_seqs], \
                                        [train_format_seqs[i] for i in train_seqs]
    valid_data, test_data, train_data = pd.DataFrame(valid_data), pd.DataFrame(test_data), pd.DataFrame(train_data)
    train_data.to_csv(os.path.join(relative_data_path,"data/tcr_seqs_train_df.csv"))
    valid_data.to_csv(os.path.join(relative_data_path,"data/tcr_seqs_dev_df.csv"))
    test_data.to_csv(os.path.join(relative_data_path, "data/tcr_seqs_test_df.csv"))

class EpitopeClassifierDataset(Dataset):
    """
        Dataset class for BERT tuning with the epitope specificity classification. A <CLS> token added at the beginning
        of the sequences will be used as a representation of the whole sequence, and the classification will be based
        on the hidden representation of that token.
    """
    def __init__(self, raw_path, file) -> None:
        self.data = []
        vocab = pickle.load(open(raw_path + "data/epitope2labels.bin", "rb"))
        path = raw_path+file
        self.init_dataset(path, vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seq = self.data.iloc[item, 0]
        target = self.data.iloc[item, 1]
        sample = {"seq": seq, "target": target}
        return sample

    def collate_lists(self, seq: list, label: list) -> dict:
        """ Converts each line into a dictionary. """
        collated_dataset = []
        for i in range(len(seq)):
            collated_dataset.append({"seq": str(seq[i]), "label": label[i]})
        return collated_dataset

    def calculate_stat(self, path):
        df = pd.read_csv(path, names=['input', 'loc', 'membrane'], skiprows=1)
        df = df.loc[df['membrane'].isin(["M", "S"])]

        self.nSamples_dic = df['membrane'].value_counts()

    def load_dataset(self, path):
        self.init_dataset(path)
        return self

    def init_dataset(self, path, vocab):
        df = pd.read_csv(path)
        label, sequences = [], []
        for ep, seq in df.values:
            label.append(vocab[ep])
            sequences.append(" ".join([s for s in seq]))
        assert len(sequences) == len(label)
        self.data = pd.DataFrame({"seq": sequences, "label": label})


class BertDataset(Dataset):
    """
    Loads the Dataset from the csv files passed to the parser. Extracts 15% of residues in seequences and replaces them
    with <MASK>. Also retains the masked indices - the respective indices will be used later to extract the masked
    residues and predict the missing residues.
            Inputs:
                file: csv file containing only raw sequences
                special_tokens: whether the model uses special tokens like <CLS>: (shift the mask indices in that case)
                relative_data_path: path to file
    """

    def __init__(self, file, special_tokens, relative_data_path) -> None:
        self.data = []
        self.init_dataset(os.path.join(relative_data_path,file), special_tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        seqs = self.data.iloc[item, 0]
        target = self.data.iloc[item, 1]
        target_pos = self.data.iloc[item, 2]
        sample = {"seq": seqs, "target": target, "target_pos": target_pos}
        return sample

    def collate_lists(self, seq: list, label: list, label_inds: list) -> dict:
        """ Converts each line into a dictionary. """
        collated_dataset = []
        for i in range(len(seq)):
            collated_dataset.append({"seq": str(seq[i]), "label": label[i], "label_inds": label_inds[i]})
        return collated_dataset

    def calculate_stat(self, path):
        df = pd.read_csv(path, names=['input', 'loc', 'membrane'], skiprows=1)
        df = df.loc[df['membrane'].isin(["M", "S"])]

        self.nSamples_dic = df['membrane'].value_counts()

    def load_dataset(self, path):
        self.init_dataset(path)
        return self

    def init_dataset(self, path, special_tokens):
        def create_labels(sequences):
            all_lbls, all_pos, masked_s = [], [], []
            for s in sequences:
                if type(s) == str:
                    current_s = s.split(" ")
                    cs = s.replace(" ", "")
                    no_masks = int((15 / 100) * len(cs))
                    inds = random.sample(list(range(len(cs))), no_masks)
                    lbl, lbl_pos = [], []
                    for i in inds:
                        lbl.append(cs[i])
                        if special_tokens:
                            lbl_pos.append(i + 1)
                        else:
                            lbl_pos.append(i)
                        current_s[i] = "[MASK]"
                    all_lbls.append(lbl)
                    all_pos.append(lbl_pos)
                    masked_s.append(" ".join(current_s))
            return all_lbls, all_pos, masked_s
        df = pd.read_csv(path, names=['sequences'], skiprows=1)

        seq = list(df['sequences'])
        # label = list(df['membrane'])
        label, label_inds, seq = create_labels(seq)
        assert len(seq) == len(label)
        assert len(seq) == len(label_inds)
        self.data = pd.DataFrame({"seq": seq, "label": label, "label_inds": label_inds})


class ProtBertClassifier(pl.LightningModule):
    """
    # https://github.com/minimalist-nlp/lightning-text-classification.git

    Sample model to show how to use BERT to classify sentences.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams) -> None:
        super(ProtBertClassifier, self).__init__()
        self.hparams = hparams
        self.batch_size = self.hparams.batch_size

        self.modelFolderPath = './models/ProtBert/'
        self.vocabFilePath = os.path.join(self.modelFolderPath, 'vocab.txt')

        self.extract_emb = False
        self.metric_acc = gpu_acc_metric

        # build model
        self.__download_model()

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs

    def __download_model(self) -> None:
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
        self.ProtBertBFD = BertModel.from_pretrained(self.modelFolderPath,
                                                     gradient_checkpointing=self.hparams.gradient_checkpointing)
        self.encoder_features = 1024

        # Tokenizer
        self.tokenizer = BertTokenizer(self.vocabFilePath, do_lower_case=False)

        # Label Encoder
        self.label_encoder = LabelEncoder(
            self.hparams.label_set.split(","), reserved_labels=[]
        )
        self.label_encoder.unknown_index = None
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

    @staticmethod
    def get_epitope_weights(relative_data_path):
        vdj_long_data = pd.read_csv(os.path.join(relative_data_path, "data/vdj_human_uniques_long.csv"))
        epitope2ind = pickle.load(open(os.path.join(relative_data_path ,"data/epitope2labels.bin"), "rb"))
        epitope2count = {}
        for ep in vdj_long_data['epitope'].values:
            if ep in epitope2count:
                epitope2count[ep] += 1
            else:
                epitope2count[ep] = 1
        ind_ep_2weights = {}
        n_samples = len(vdj_long_data['epitope'].values)
        n_classes = len(epitope2ind.keys())
        for epitope, ind in epitope2ind.items():
            ind_ep_2weights[ind] = n_samples / (n_classes * epitope2count[epitope])
        ordered_weights = []
        for ind in range(n_classes):
            ordered_weights.append(ind_ep_2weights[ind])
        return ordered_weights

    def __build_loss(self):
        """ Initializes the loss function/s. """
        if self.hparams.tune_epitope_specificity:
            weights = self.get_epitope_weights(self.hparams.relative_data_path)
            self._loss = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        else:
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

    def forward(self, input_ids, token_type_ids, attention_mask, target_positions=None, return_embeddings=False):
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
        # return {"logits": self.classification_head(prediction_embeddings)}
        out = self.classification_head(prediction_embeddings)

        # return self.classification_head(prediction_embeddings)
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
        # return self._loss(predictions["logits"], torch.tensor(targets["labels"], device=predictions["logits"].device))
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
        # y_hat = model_out["logits"]
        y_hat = model_out
        labels_hat = torch.argmax(y_hat, dim=1)
        # labels_hat = y_hat
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
        # y_hat = model_out["logits"]
        y_hat = model_out

        labels_hat = torch.argmax(y_hat, dim=1)
        # labels_hat = y_hat
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
        # optimizer = Lamb(parameters, lr=self.hparams.learning_rate, weight_decay=0.01)
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    def __retrieve_dataset(self, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        if self.hparams.tune_epitope_specificity:
            if train:
                return EpitopeClassifierDataset(hparams.relative_data_path, hparams.train_csv)
            elif val:
                return EpitopeClassifierDataset(hparams.relative_data_path, hparams.dev_csv)
            elif test:
                return EpitopeClassifierDataset(hparams.relative_data_path, hparams.test_csv)
            else:
                print('Incorrect dataset split')
        else:
            if train:
                return BertDataset(hparams.train_csv, hparams.special_tokens, hparams.relative_data_path)
            elif val:
                return BertDataset(hparams.dev_csv, hparams.special_tokens, hparams.relative_data_path)
            elif test:
                return BertDataset(hparams.test_csv, hparams.special_tokens, hparams.relative_data_path)
            else:
                print('Incorrect dataset split')

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._dev_dataset = self.__retrieve_dataset(train=False, test=False)
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @classmethod
    def add_model_specific_args(
            cls, parser: HyperOptArgumentParser
    ) -> HyperOptArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters.
        :param parser: HyperOptArgumentParser obj
        Returns:
            - updated parser
        """
        parser.opt_list(
            "--max_length",
            default=1536,
            type=int,
            help="Maximum sequence length.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=5e-6,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-5,
            type=float,
            help="Classification head learning rate.",
        )
        parser.opt_list(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
            tunable=True,
            options=[0, 1, 2, 3, 4, 5],
        )
        # Data Args:
        parser.add_argument(
            "--label_set",
            default="M,S",
            type=str,
            help="Classification labels set.",
        )
        parser.add_argument(
            "--train_csv",
            default="data/tcr_seqs_train_df.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="data/tcr_seqs_dev_df.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            default="data/tcr_seqs_test_df.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        parser.add_argument(
            "--gradient_checkpointing",
            default=False,
            action="store_true",
            help="Enable or disable gradient checkpointing which use the cpu memory \
                with the gpu memory to store the model.",
        )
        return parser


def setup_testube_logger(save_dir="experiments") -> TestTubeLogger:
    """ Function that sets the TestTubeLogger to be used. """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    save_dir = "experiments/"
    return TestTubeLogger(
        save_dir=save_dir,
        version=dt_string,
        name="lightning_logs",
    )

def parse_arguments_and_retrieve_logger(save_dir="experiments"):
    """
        Function for parsing all arguments
    """
    logger = setup_testube_logger(save_dir=save_dir)
    parser = HyperOptArgumentParser(
        strategy="random_search",
        description="Minimalist ProtBERT Classifier",
        add_help=True,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    parser.add_argument(
        "--create_data",
        default=False,
        action="store_true",
        help="Create data for bert fine-tuning out of emerson-long and tcrb files",
    )
    parser.add_argument(
        "--special_tokens",
        default=False,
        action="store_true",
        help="Tune the ProtBert ot the epitope classification task"
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_acc", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=2,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=64,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )
    parser.add_argument(
        "--distributed_backend",
        default="ddp",
        type=str,
        help=(
            "Parallelization method for multi-gpu training. Default ddp is strongly advised."
        ),
    )
    parser.add_argument(
        "--no_sequences",
        default=3 * 10 ** 6,
        type=int,
        help="Number of sequences to be used in training and testing. Only used if create_data is true and"
             " tcrb_only is false."
    )

    # gpu/tpu args
    parser.add_argument("--gpus", type=int, default=1, help="How many gpus")
    parser.add_argument("--tpu_cores", type=int, default=None, help="How many tpus")
    parser.add_argument(
        "--val_percent_check",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )
    parser.add_argument("--only_tcrb", default=False, action="store_true", help="Create and train on only the tcrb files")

    # mixed precision
    parser.add_argument("--precision", type=int, default="32", help="full precision or mixed precision mode")
    parser.add_argument("--amp_level", type=str, default="O1", help="mixed precision type")
    parser.add_argument("--tune_epitope_specificity", default=False, action="store_true")
    parser.add_argument("--embedding_save_name",default="some_emb", type=str)
    parser.add_argument("--add_long_aa",default=-1, type=int)
    parser.add_argument("--relative_data_path",default="./", type=str)

    # each LightningModule defines arguments relevant to it
    parser = ProtBertClassifier.add_model_specific_args(parser)
    hparams = parser.parse_known_args()[0]
    return hparams, logger


def create_tuning_data(hparams):
    """
        When called without parameter create_data, this method only sets the appropriate file names for the chosen training
        scheme (epitope fine tuning or further bert-like tuning)
    """
    if hparams.tune_epitope_specificity and not hparams.special_tokens:
        print("WARNING!!! Called the training with epitope classifier fine tuning but did not set "
              "the special_tokens parameter. Setting it to true atuomatically...")
        hparams.special_tokens = True
    if hparams.create_data:
        if hparams.tune_epitope_specificity:
            hparams.test_csv, hparams.train_csv, hparams.dev_csv = "data/epitope_seq_test.csv", "data/epitope_seq_train.csv", "data/epitope_seq_valid.csv"
            create_epitope_tuning_files(hparams.relative_data_path)
        else:
            create_bert_further_tuning_files(hparams.relative_data_path)


    if hparams.gradient_checkpointing and hparams.distributed_backend == "ddp":
        print("!!!ERROR!!! When using ddp as the distributed backend method, gradient checkpointing "
              "does not work. Exiting...")
        exit(1)

if __name__=="__main__":

    hparams, logger = parse_arguments_and_retrieve_logger(save_dir="experiments")
    create_tuning_data(hparams)
    if hparams.gradient_checkpointing and hparams.distributed_backend == "ddp":
        # gradient checkpoint does not work with ddp, which is necessary for multi-gpu training
        print("!!!ERROR!!! When using ddp as the distributed backend method, gradient checkpointing "
              "does not work. Exiting...")
        exit(1)
    model = ProtBertClassifier(hparams)

    if hparams.nr_frozen_epochs == 0:
        model.freeze_encoder()
        model.unfreeze_encoder()

    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )

    ckpt_path = os.path.join(
        logger.save_dir,
        logger.name,
        f"version_{logger.version}",
        "checkpoints",
    )
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path + "/" + "{epoch}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        period=1,
        mode=hparams.metric_mode,
    )

    trainer = Trainer(
        gpus=hparams.gpus,
        tpu_cores=hparams.tpu_cores,
        logger=logger,
        early_stop_callback=early_stop_callback,
        distributed_backend=hparams.distributed_backend,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        val_percent_check=hparams.val_percent_check,
        checkpoint_callback=checkpoint_callback,
        precision=hparams.precision,
        amp_level=hparams.amp_level,
        deterministic=True
    )
    trainer.fit(model)
