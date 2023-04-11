# check pytorch version
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
# pytorch mlp for binary classification
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

class CSVDataset(Dataset):
    # load the csv file as a dataframe
    df = pd.read_csv(path, header=[0],
                  index_col=0,
                  parse_dates=True
                  )
    df.index = pd.to_datetime(df.index,
                              utc=True
                              ).tz_convert(tz="Europe/Berlin")

    # store the inputs and outputs
    self.X = df.values[:, :-1]
    self.y = df.values[:, -1]
    # ensure input data is floats
    self.X = self.X.astype('float32')
    # label encode target and ensure the values are floats
    self.y = LabelEncoder().fit_transform(self.y)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
    # prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=96, shuffle=False)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl, train, test

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

df_gen = pd.DataFrame()
df_gen = pd.read_csv("./data/entsoe_gen.csv",
                   header=[0],
                   index_col=0,
                   decimal=",",
                   low_memory=False,
                  ).drop(
    ['Fossil Gas.1',
     'Fossil Oil.1',
     'Hydro Water Reservoir.1',
     'Nuclear.1',
     'Other renewable.1',
     'Solar.1',
     'Wind Onshore.1'],
    axis=1,
)

df_gen.drop(index=df_gen.index[0], axis=0, inplace=True)

df_gen.index = pd.to_datetime(df_gen.index,
                                    utc=True
                                    ).tz_convert(tz="Europe/Berlin")