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
  
  n = len(X_train)
  index = torch.randint(n, (int(n * prop), ))
  
  return x[index], y[index]
