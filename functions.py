#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 13:41:55 2019

@author: will
"""
import tensorflow as tf
import numpy as np
import pickle

class DataGenerator(tf.keras.utils.Sequence):
    # constants
    tot_len = 803
    y_len = 60
    year_len = 366
    value_path = '/home/will/Desktop/kaggle/WT/Data/value.npy'
    IsMissing_path = '/home/will/Desktop/kaggle/WT/Data/IsMissing.npy'
    start_index_path = '/home/will/Desktop/kaggle/WT/Data/start_index.npy'
    date_index_path = '/home/will/Desktop/kaggle/WT/Data/date_index'
    page_path = '/home/will/Desktop/kaggle/WT/Data/Page_X.npy'
    
    def __init__(self, batchSize,normalize,x_len=64):
        'load from disk'
        start_index = np.load(DataGenerator.start_index_path)
        values = np.load(DataGenerator.value_path)
        IsMissing = np.load(DataGenerator.IsMissing_path)
        page = np.load(DataGenerator.page_path)
        with open(DataGenerator.date_index_path, "rb") as input_file:
            date_index = pickle.load(input_file)
        
        self.batchSize = batchSize
        self.filter_index = (DataGenerator.tot_len - start_index) > (DataGenerator.year_len + 2*DataGenerator.y_len) # too few lenth for training
        self.start_index = start_index[self.filter_index] # numpy array with starting index for each row
        self.IsMissing = IsMissing[self.filter_index] # same shape as values
        self.train = values[self.filter_index,:-DataGenerator.y_len] # remove most recent data for val
        self.page = page[self.filter_index]
        self.x_len = x_len # lenth of lookback period
        self.end_index = DataGenerator.tot_len - 2*DataGenerator.y_len - x_len - 1
        self.values = values # np.array of shape (N, tot_len)
        self.date_index = date_index # pd.timestamp of length tot_len
        self.normalize = normalize
        if normalize:
            self.mu = np.mean(self.train,1)
            self.std = np.std(self.train,1)
        else:
            self.mu = None
            self.std = None
        self.on_epoch_begin()

    def __len__(self):
        'Denotes the number of batches per epoch.'
        return int(self.train.shape[0]/self.batchSize)

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_index = self.random_index[index*self.batchSize:(index+1)*self.batchSize]
        page = self.page[batch_index]
        random_start = (self.start_index[batch_index] + (self.end_index-self.start_index[batch_index])*np.random.rand(self.batchSize)).astype(np.int32)
        return page.astype(np.float32)

    def on_epoch_begin(self):
        self.random_index = np.random.permutation(self.train.shape[0])
            
    @staticmethod
    def __create2(img_list):
        len_ = len(img_list)
        if len_ <= 2:
            return img_list
        else:
            np.random.shuffle(img_list)
            return img_list[:2]
        
    def __data_generation(self, indexes):
        imgs_list = [[np.load(img) for img in group] for group in indexes]
        imgs_list = [[self.transFun(group[0])[:,:,np.newaxis],self.transFun(group[0])[:,:,np.newaxis]] if len(group)==1 
                      else [self.transFun(group[0])[:,:,np.newaxis],self.transFun(group[1])[:,:,np.newaxis]] for group in imgs_list]
        X1,X2 = list(zip(*imgs_list))
        r = np.random.randint(1,self.HalfBatch)
        X1,X2 = list(X1),list(X2)
        X1.extend(X1)
        X2.extend([X2[(i+r)%self.HalfBatch] for i in range(self.HalfBatch)])
        return np.array(X1),np.array(X2)
    