import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class Data(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/test_all.csv')
        data = data.values

        chunks = np.split(data, data.shape[0] // win_size, axis=0)
        np.random.shuffle(chunks)
        data = np.vstack(chunks)

        train_data = data[:int(len(data) * 0.7), :-1]
        train_data = np.nan_to_num(train_data)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = data[int(len(data) * 0.7):, :-1] #int(len(data) * 0.05) int(len(data) * 0.95)
        test_data = np.nan_to_num(test_data)
        self.train = train_data
        self.test = self.scaler.transform(test_data)
        self.val = self.test

        self.train_labels = data[:int(len(data) * 0.7), -1]
        self.test_labels = data[int(len(data) * 0.7):, -1]
        # print(train_data.shape)
        # test_data = pd.read_csv(data_path + '/test.csv')
        # self.labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]
        # self.train_labels = self.labels[:int(len(data) * 0.7), :]
        # self.test_labels = self.labels[int(len(data) * 0.7):, :]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(
                self.train_labels[index:index + self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=20, step=20, mode='train', dataset='KDD'):
    dataset = Data(data_path, win_size, 20, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
