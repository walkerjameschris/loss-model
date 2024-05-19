import yaml
import numpy as np
import pandas as pd
import torch

def load_yaml(x):
  '''Proper YAML loading'''

  with open(x, 'r') as file:
    return yaml.safe_load(file)

def to_numpy(x):
  '''Detaches a tensor and converts to NumPy'''

  return x.clone().detach().numpy().reshape(-1)

def to_tensor(x):
  '''Converts an array to properly shaped tensor'''
  
  x = torch.tensor(x, dtype=torch.float)

  if (len(x.shape) == 1):
    x = x.reshape(-1, 1)
    
  return x

def to_disk(x, y, label):
  '''Writes actual and predicted to disk'''
  
  data = pd.DataFrame({
    ".pred": to_numpy(x),
    "gross_loss": to_numpy(y),
    "population": label
  })
  
  data.to_csv(label, index=False)
  
def get_tmr(pred, real):
  '''Computes TMR over actuals and predicted'''

  real = to_numpy(real)
  pred = to_numpy(pred)
  
  real = real >= np.percentile(real, 95)
  pred = pred >= np.percentile(pred, 95)
  
  # Compute match
  return round(100 * sum(real * pred) / sum(real), 2)

def load_data(x, y, prop=0.2):
  '''Sampling dataloader utility'''
  
  n = len(x)
  index = torch.randint(n, (int(n * prop), ))
  
  return x[index], y[index]

def get_active(x):
  '''Gets PyTorch activations'''

  if x == "ReLU":
    return torch.nn.ReLU()
  if x == "LeakyReLU":
    return torch.nn.LeakyReLU()

def fit_model(X_train,
              y_train,
              n_hidden,
              dropout,
              active_fn,
              learn_rate=0.01,
              n_epochs=3000):
  '''Common utility for fitting a PyTorch model'''

  print(
    f'\nFitting Model: {n_hidden} Hidden,',
    f'{dropout} Drop Rate, {active_fn}\n'
  )

  _, n_predictors = X_train.shape
  active_fn = get_active(active_fn)

  model = torch.nn.Sequential(
    torch.nn.Dropout(dropout),
    torch.nn.Linear(n_predictors, n_hidden),
    active_fn,
    torch.nn.Linear(n_hidden, n_hidden),
    active_fn,
    torch.nn.Linear(n_hidden, n_hidden),
    active_fn,
    torch.nn.Linear(n_hidden, 1)
  )

  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

  for epoch in range(n_epochs):
    
    model.train()
    x, y = load_data(X_train, y_train)
    
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    
    if epoch % 500 == 0:
      tmr = get_tmr(model(X_train), y_train)
      epoch = str(epoch).zfill(len(str(n_epochs)))
      print(f'Epoch: {epoch} TMR: {tmr}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()
  
  return model
