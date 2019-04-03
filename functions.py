#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:03:44 2019

@author: will
"""
import numpy as np
from torch.utils.data import Dataset
from torch import nn
import torch


''' data generator '''

class SequenceGen(Dataset):
    def __init__(self,long_value,long_normal_value,start_index,stds,means,
                 end_index=803-60-60,channelFirst=True,random=True,useMean=False):
        # random == True for training, and False for validation
        self.long_value = long_value
        self.long_normal_value = long_normal_value
        self.start_index = start_index
        self.stds = stds
        self.means = means
        self.end_index = end_index
        self.channelFirst = channelFirst
        self.random = random
        self.useMean = useMean
        
    def __len__(self):
        return self.long_value.shape[0]

    def __getitem__(self, idx):
        r = np.random.randint(self.start_index[idx],self.end_index) if self.random else self.end_index+60
        X = np.stack([self.long_normal_value[idx,r-120:r],
                      self.long_normal_value[idx,r-242:r-122],
                      self.long_normal_value[idx,r-425:r-305],
                        ],0 if self.channelFirst else 1)
        std = self.stds[idx]
        mean = self.means[idx] if self.useMean else self.long_value[idx,r-1:r]
        y = self.long_value[idx,r:r+60]
        return X,std,mean,y

class SequenceGenNormalized(Dataset):
    def __init__(self,long_normal_value,start_index,
                 end_index=803-60-60,channelFirst=True,random=True):
        # random == True for training, and False for validation
        self.long_normal_value = long_normal_value
        self.start_index = start_index
        self.end_index = end_index
        self.channelFirst = channelFirst
        self.random = random

    def __len__(self):
        return self.long_normal_value.shape[0]

    def __getitem__(self, idx):
        r = np.random.randint(self.start_index[idx],self.end_index) if self.random else self.end_index+60
        X = np.stack([self.long_normal_value[idx,r-120:r],
                      self.long_normal_value[idx,r-242:r-122],
                      self.long_normal_value[idx,r-425:r-305],
                        ],0 if self.channelFirst else 1)
        y = self.long_normal_value[idx,r:r+60]
        return X,y

class SequenceGenLong(Dataset):
    def __init__(self,long_value,long_normal_value,start_index,stds,means,
                 end_index=803-60-60,channelFirst=True,random=True,useMean=False):
        # random == True for training, and False for validation
        self.long_value = long_value
        self.long_normal_value = long_normal_value
        # -50 as we need look back 365 instead of 420
        self.start_index = start_index-50
        self.stds = stds
        self.means = means
        self.end_index = end_index
        self.channelFirst = channelFirst
        self.random = random
        self.useMean = useMean
        
    def __len__(self):
        return self.long_value.shape[0]

    def __getitem__(self, idx):
        r = np.random.randint(self.start_index[idx],self.end_index) if self.random else self.end_index+60
        X = np.expand_dims(self.long_normal_value[idx,r-366:r],0 if self.channelFirst else 1)
        std = self.stds[idx]
        mean = self.means[idx] if self.useMean else self.long_value[idx,r-1:r]
        y = self.long_value[idx,r:r+60]
        return X,std,mean,y

class SequenceGenNormalizedLong(Dataset):
    def __init__(self,long_normal_value,start_index,
                 end_index=803-60-60,channelFirst=True,random=True):
        # random == True for training, and False for validation
        self.long_normal_value = long_normal_value
        # -50 as we need look back 365 instead of 420
        self.start_index = start_index - 50
        self.end_index = end_index
        self.channelFirst = channelFirst
        self.random = random

    def __len__(self):
        return self.long_normal_value.shape[0]

    def __getitem__(self, idx):
        r = np.random.randint(self.start_index[idx],self.end_index) if self.random else self.end_index+60
        X = np.expand_dims(self.long_normal_value[idx,r-366:r],0 if self.channelFirst else 1)
        y = self.long_normal_value[idx,r:r+60]
        return X,y
    
class CNN_RNN2seq(nn.Module):
    def __init__(self,conv,rnn,linear):
        super().__init__()
        self.conv = conv 
        self.rnn = rnn
        self.linear = linear
        
    def forward(self,X,std,mean):
        X = self.conv(X).transpose(1,2)
        X,_ = self.rnn(X)
        X = X[:,-60:]
        X = self.linear(X).squeeze(2)
        X = (X*std)+mean
        return X
    
class CNN_RNN2seq_normalized(nn.Module):
    def __init__(self,conv,rnn,linear):
        super().__init__()
        self.conv = conv 
        self.rnn = rnn
        self.linear = linear
        
    def forward(self,X):
        X = self.conv(X).transpose(1,2)
        X,_ = self.rnn(X)
        X = X[:,-60:]
        X = self.linear(X).squeeze(2)
        return X    

class CNN_RNN2seq_logNormalized(nn.Module):
    def __init__(self,conv,rnn,linear):
        super().__init__()
        self.conv = conv 
        self.rnn = rnn
        self.linear = linear
        
    def forward(self,X,std,mean):
        X = self.conv(X).transpose(1,2)
        X,_ = self.rnn(X)
        X = X[:,-60:]
        X = self.linear(X).squeeze(2)
        X = torch.exp((X*std)+mean) - 1
        return X
    
def loss_func_generator(distanceFun):
    def loss_func(model,data):
        X,std,mean,y = data
        yhat = model(X,std,mean)
        loss = distanceFun(yhat,y)
        return loss
    return loss_func    

class CNN_RNN2seq_NCL(nn.Module):
    # input/oputput with shape (batch, channel, length) for RNN and CNN
    # rnn needs to be instance of GRU_NCL
    def __init__(self,seq_model,linear,convert=None):
        # convert maps N,C,L to N,C,60
        super().__init__()
        self.seq_model = seq_model 
        self.linear = linear
        self.convert = convert
        
    def forward(self,X,std,mean):
        X = self.seq_model(X)
        X = X[:,:,-60:] if self.convert is None else self.convert(X).transpose(1,2)
        X = self.linear(X).squeeze(2)
        X = (X*std)+mean
        return X
    
class CNN_RNN2seq_normalized_NCL(nn.Module):
    def __init__(self,seq_model,linear,convert=None):
        super().__init__()
        self.seq_model = seq_model
        self.linear = linear
        self.convert = convert
        
    def forward(self,X):
        X = self.seq_model(X)
        X = X[:,:,-60:] if self.convert is None else self.convert(X).transpose(1,2)
        X = self.linear(X).squeeze(2)
        return X    

class CNN_RNN2seq_logNormalized_NCL(nn.Module):
    def __init__(self,seq_model,linear,convert=None):
        super().__init__()
        self.seq_model = seq_model 
        self.linear = linear
        self.convert = convert
        
    def forward(self,X,std,mean):
        X = self.seq_model(X)
        X = X[:,:,-60:] if self.convert is None else self.convert(X).transpose(1,2)
        X = self.linear(X).squeeze(2)
        X = torch.exp((X*std)+mean) - 1
        return X

def loss_func_generator_normalized(distanceFun):
    def loss_func(model,data):
        X,y = data
        yhat = model(X)
        loss = distanceFun(yhat,y)
        return loss
    return loss_func  
    
def SMAPE(y,yhat):
    summ = torch.abs(y) + torch.abs(yhat) + 1e-3
    return 200*torch.mean(torch.abs(y-yhat)/summ)
    
def SMAPE_np(y,yhat):
    summ = np.abs(y) + np.abs(yhat) + 1e-3
    return 200*np.mean(np.abs(y-yhat)/summ)    
    
    
    
    
    
    
    
    


