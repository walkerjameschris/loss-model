#### Setup ####

# deep_learning.py
# Chris Walker

import sys
import yaml
import torch
import numpy as np
import pandas as pd

torch.manual_seed(12345)

with open('src/dials.yaml', 'r') as file:
  dials = yaml.safe_load(file)

train = pd.read_csv('train')
test = pd.read_csv('test')

learn_rate = dials['torch_rate']
n_epochs = dials['torch_epochs']
n_hidden = dials['torch_hidden']
dropout = dials['torch_dropout']
target = 'gross_loss'

def to_numpy(x):
  
  # Make a copy, detach, and convert
  x = x.clone().detach()
  return x.numpy().reshape(-1)

def to_tensor(x):
  
  # Convert to tensor
  x = torch.tensor(x, dtype=torch.float)
  
  # Handle singular vectors
  if (len(x.shape) == 1):
    x = x.reshape(-1, 1)
    
  return x

def to_disk(x, y, label):
  
  data = pd.DataFrame({
    ".pred": to_numpy(x),
    "gross_loss": to_numpy(y),
    "population": label
  })
  
  data.to_csv(label, index=False)
  
def get_tmr(pred, real):
  
  # Convert to NumPy
  real = to_numpy(real)
  pred = to_numpy(pred)
  
  # Determine ranks
  real = real >= np.percentile(real, 95)
  pred = pred >= np.percentile(pred, 95)
  
  # Compute match
  return round(100 * sum(real * pred) / sum(real), 2)

def load_data(x, y, prop=0.2):
  
  # Sample some proportion of X
  n = len(X_train)
  index = torch.randint(n, (int(n * prop), ))
  
  return x[index], y[index]

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
  x, y = load_data(X_train, y_train)
  
  y_pred = model(x)
  loss = loss_fn(y_pred, y)
  
  # Epoch loss report
  if epoch % 100 == 0:
    epoch = str(epoch).zfill(len(str(n_epochs)))
    tmr = get_tmr(model(X_train), y_train)
    print(f"Epoch: {epoch} MSE: {loss} TMR: {tmr}")
  
  # Zero gradients, backpropagate, and tune
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  model.eval()

# Save out to disk
to_disk(model(X_train), y_train, "train")
to_disk(model(X_test), y_test, "test")
