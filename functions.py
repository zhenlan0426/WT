#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:03:44 2019

@author: will
"""
import numpy as np
from torch.utils.data import Dataset
from torch import nn


class SequenceGen(Dataset):
    def __init__(self,long_value,long_normal_value,start_index,stds,means,
                 end_index=803-60-60,channelFirst=True,random=True):
        # random == True for training, and False for validation
        self.long_value = long_value
        self.long_normal_value = long_normal_value
        self.start_index = start_index
        self.stds = stds
        self.means = means
        self.end_index = end_index
        self.channelFirst = channelFirst
        self.random = random

    def __len__(self):
        return self.long_value.shape[0]

    def __getitem__(self, idx):
        r = np.random.randint(self.start_index[idx],self.end_index) if self.random else self.end_index+60
        X = np.stack([self.long_normal_value[idx,r-120:r],
                      self.long_normal_value[idx,r-242:r-122],
                      self.long_normal_value[idx,r-425:r-305],
                        ],0 if self.channelFirst else 1)
        std = self.stds[idx]
        mean = self.means[idx]
        y = self.long_value[idx,r:r+60]
        return X,std,mean,y


class CNN_RNN2seq(nn.Module):
    def __init__(self,conv,rnn,linear):
        super(CNN_RNN2seq, self).__init__()
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
    
def loss_func_generator(distanceFun):
    def loss_func(model,data):
        X,std,mean,y = data
        yhat = model(X,std,mean)
        loss = distanceFun(yhat,y)
        return loss
    return loss_func    
    



