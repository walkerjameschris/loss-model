#### Setup ####

# deep_learning.py
# Chris Walker

import sys
import yaml
import torch
import numpy as np
import pandas as pd

with open('src/dials.yaml', 'r') as file:
  dials = yaml.safe_load(file)

train = pd.read_csv('train')
test = pd.read_csv('test')

loss_rate = dials['torch_rate']
n_epochs = dials['torch_epochs']
n_hidden = dials['torch_hidden']
target = 'gross_loss'

def to_tensor(x):
  
  # Convert to tensor
  x = torch.tensor(x, dtype=torch.float)
  
  # Handle singular vectors
  if (len(x.shape) == 1):
    x = x.reshape(-1, 1)
    
  return x

def to_disk(x, y, label):
  
  data = pd.DataFrame({
    ".pred": x.detach().reshape(-1).numpy(),
    "gross_loss": y.reshape(-1).numpy(),
    "population": label
  })
  
  data.to_csv(label, index=False)

#### Convert to Tensors ####

y_train = to_tensor(train[target])
y_test = to_tensor(test[target])

train = train.drop(columns=target)
test = test.drop(columns=target)

X_train = to_tensor(train.values)
X_test = to_tensor(test.values)

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
  torch.nn.Linear(n_hidden, n_hidden),
  torch.nn.ReLU(),
  torch.nn.Linear(n_hidden, 1)
)

# MSE loss is common for regression
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=loss_rate)

for epoch in range(n_epochs):
  
  # Switch to train mode and compute loss
  model.train()
  y_pred = model(X_train)
  loss = loss_fn(y_pred, y_train)
  
  # Epoch loss report
  print(f"Epoch {epoch} MSE: {loss}")
  
  # Zero gradients, backpropagate, and tune
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  model.eval()

# Save out to disk
to_disk(model(X_train), y_train, "train")
to_disk(model(X_test), y_test, "test")
