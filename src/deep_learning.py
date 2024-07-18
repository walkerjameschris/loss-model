#### Setup ####

# deep_learning.py
# Chris Walker

import yaml
import utils
from itertools import product
import numpy as np
import pandas as pd
import torch

torch.manual_seed(12345)

dials = utils.load_yaml('src/dials.yaml')

train = pd.read_csv('train')
test = pd.read_csv('test')

active_fun = dials['torch_fun']
n_hidden = dials['torch_hidden']
dropout_pct = dials['torch_dropout']
tune = dials['tune_torch']
target = 'gross_loss'

#### Convert to Tensors ####

y_train = utils.to_tensor(train[target])
y_test = utils.to_tensor(test[target])

train = train.drop(columns=target)
test = test.drop(columns=target)

X_train = utils.to_tensor(train.values)
X_test = utils.to_tensor(test.values)

#### Tune the Model ####

if tune:

  num_hidden = dials['num_hidden']
  dropout = dials['dropout']
  active_fn = dials['active_fn']

  hide_log = []
  drop_log = []
  active_log = []

  for n, d, a in product(num_hidden, dropout, active_fn):

    model = utils.fit_model(X_train, y_train, n, d, a)

    hide_log.append(n)
    drop_log.append(d)
    active_log.append(a)

  pd.DataFrame({
    'n_hidden': hide_log,
    'dropout': drop_log,
    'active_fn': active_log
  }).to_csv('data/tune_torch.csv', index=False)

#### Fit and Save ####

model = utils.fit_model(
  X_train,
  y_train,
  n_hidden,
  dropout_pct,
  active_fun
)

# Save out to disk
utils.to_disk(model(X_train), y_train, 'train')
utils.to_disk(model(X_test), y_test, 'test')
