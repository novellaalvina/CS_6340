# models.py

import numpy as np
import argparse
import json
import time
import random
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
from transformer import PositionalEncoding

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, d_model, num_positions, nhead, num_layers, dropout, vocab_index: Indexer):
        
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_positions = num_positions
        self.nhead = nhead
        self.dropout = dropout
        self.vocab_index = vocab_index

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=self.nhead, dropout=self.dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer, num_layers=self.num_layers)
        
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding =  PositionalEncoding(self.d_model, self.num_positions)
        
        self.W = nn.Linear(self.d_model, self.vocab_size)
        nn.init.xavier_uniform_(self.W.weight)
        
    def forward(self, indices):
        
        # embedding
        input_embedding = self.embedding(indices)

        # positional encoding
        input_positional_encoding = self.positional_encoding.forward(input_embedding)

        # masking 
        mask = torch.triu(torch.ones(self.d_model, self.d_model), diagonal=1).bool()
        
        # mask filling with -infinity
        input_mask = mask.masked_fill(mask, float('-inf'))
        
        # transformer
        transformer_encoder_output = self.transformer_encoder.forward(src=(input_positional_encoding), mask=input_mask, is_causal=True)

        # linear and log softmax
        log_softmax = nn.functional.log_softmax(input=self.W(transformer_encoder_output), dim=-1, dtype=torch.float64)
        
        return log_softmax
                
    def get_next_char_log_probs(self, context):
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        indices = []
        for char in context:
            char_index = self.vocab_index.index_of(char)
            indices.append(char_index)
        log_probs = self.forward(torch.tensor(indices))
                
        # get probs only for next char
        log_probs = log_probs[-1, :]
        log_probs = log_probs.detach().numpy()
        
        self.transformer_encoder.eval()
        
        return np.float64(log_probs)

    def get_log_prob_sequence(self, next_chars, context):
        
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        
        if (len(context) == 0):
            context = " "
        
        context_log_probs = 0.0
        for char in next_chars:
            char_index = self.vocab_index.index_of(char)
            char_log_prob = self.get_next_char_log_probs(context)   
            context_log_probs += char_log_prob[char_index] 
            context += char
            
        self.transformer_encoder.eval()
        
        return context_log_probs

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """  
    vocab_size = len(vocab_index)
    d_model = 500
    chunk_size = 200
    num_positions = 500
    nhead = 2
    num_layers = 10
    dropout = 0.1
    
    train_context = []
    gold = []
    for i in range(100, d_model-1):
        train_context.append(train_text[i*chunk_size:(i+1)*chunk_size])
        gold.append(train_text[i*chunk_size+1:(i+1)*chunk_size+1])
    
    model = NeuralLanguageModel(vocab_size, d_model, num_positions, nhead, num_layers, dropout, vocab_index)
    model.transformer_encoder.zero_grad()
    model.transformer_encoder.train()
    optimizer = optim.Adam(model.transformer_encoder.parameters(), lr=1e-4)

    num_epochs = 3
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train_context))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        
        for ex_idx in ex_idxs:
            
            # indexed context
            indexed_chunked_input = [vocab_index.index_of(char) for char in train_context[ex_idx]]
            indexed_chunked_gold = [vocab_index.index_of(char) for char in gold[ex_idx]]
            log_probs = model.forward(torch.tensor(indexed_chunked_input))
            loss = loss_fcn(log_probs[0], torch.tensor(indexed_chunked_gold[0]))
            
            for i in range(1,chunk_size):
                loss += loss_fcn(log_probs[i], torch.tensor(indexed_chunked_gold[i])) 
            
            loss = torch.divide(loss, chunk_size)
            model.transformer_encoder.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()            

    model.transformer_encoder.eval()
    
    return model