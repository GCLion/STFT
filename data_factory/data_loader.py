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
    def __init__(self, data, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        data = data.values
        print(data.shape)

        chunks = np.split(data, data.shape[0] // win_size, axis=0)
        np.random.shuffle(chunks)
        data = np.vstack(chunks)

        tar_data = data[:, :-1]
        tar_data = np.nan_to_num(tar_data)
        self.scaler.fit(tar_data)
        self.tar = self.scaler.transform(tar_data)
        self.tar_labels = data[:, -1]
        print(f"{mode}:", self.tar.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.tar.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.tar.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.tar.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.tar.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.tar[index:index + self.win_size]), np.float32(
                self.tar_labels[index:index + self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.tar[index:index + self.win_size]), np.float32(
                self.tar_labels[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.tar[index:index + self.win_size]), np.float32(
                self.tar_labels[index:index + self.win_size])
        else:
            return np.float32(self.tar[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.tar_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, data_path_, batch_size, win_size=19, step=19, mode='all', dataset='KDD'):
    data = pd.read_csv(data_path)
    data_ = pd.read_csv(data_path_)
    train_dataset = Data(data, win_size, win_size, "train")
    val_dataset = Data(data_, win_size, win_size, "val")
    test_dataset = Data(data_, win_size, win_size, "test")

    shuffle = False
    train_data_loader = DataLoader(dataset=train_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    val_dataset_loader = DataLoader(dataset=val_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    test_dataset_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    
    return train_data_loader, val_dataset_loader, test_dataset_loader
