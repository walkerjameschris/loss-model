#### Setup ####

# deep_learning.py
# Chris Walker

import yaml
import utils
import numpy as np
import pandas as pd
import torch

torch.manual_seed(12345)

dials = utils.load_yaml('src/dials.yaml', 'r')

train = pd.read_csv('train')
test = pd.read_csv('test')

learn_rate = dials['torch_rate']
n_epochs = dials['torch_epochs']
n_hidden = dials['torch_hidden']
dropout = dials['torch_dropout']
target = 'gross_loss'

#### Convert to Tensors ####

y_train = utils.to_tensor(train[target])
y_test = utils.to_tensor(test[target])

train = train.drop(columns=target)
test = test.drop(columns=target)

X_train = utils.to_tensor(train.values)
X_test = utils.to_tensor(test.values)

_, n_predictors = X_train.shape

#### Define the Model ####

# Define neural network with hidden layers
model = torch.nn.Sequential(
  torch.nn.Linear(n_predictors, n_hidden),
  torch.nn.ReLU(),
  torch.nn.Linear(n_hidden, n_hidden),
  torch.nn.ReLU(),
  torch.nn.Linear(n_hidden, n_hidden),
  torch.nn.ReLU(),
  torch.nn.Linear(n_hidden, 1),
  torch.nn.Dropout(dropout)
)

# MSE loss is common for regression
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

for epoch in range(n_epochs):
  
  # Switch to train mode and compute loss
  model.train()
  
  # Use dataloader
  x, y = utils.load_data(X_train, y_train)
  
  y_pred = model(x)
  loss = loss_fn(y_pred, y)
  
  # Epoch loss report
  if epoch % 100 == 0:
    epoch = str(epoch).zfill(len(str(n_epochs)))
    tmr = utils.get_tmr(model(X_train), y_train)
    print(f'Epoch: {epoch} MSE: {loss} TMR: {tmr}')
  
  # Zero gradients, backpropagate, and tune
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  model.eval()

# Save out to disk
utils.to_disk(model(X_train), y_train, 'train')
utils.to_disk(model(X_test), y_test, 'test')
