import os
from tqdm import tqdm
import pandas as pd
import numpy as np

import selberai.data.download_data as download_data

def load(name: str, sample_only=False, path_to_data=None, token=None) -> (
  np.array, np.array, np.array):
  """
  """
  
  # set standard path to data if not provided
  if path_to_data is None:
    path_to_data = 'datasets/{}/processed/'.format(name)
  
  # list directory of dataset
  dir_cont = set(os.listdir(path_to_data))
  
  # check if dataset available
  if ('training' in dir_cont and 'testing' in dir_cont and 
    'validation' in dir_cont):
    dir_cont_train = os.listdir(path_to_data+'training/')
    dir_cont_val = os.listdir(path_to_data+'validation/')
    dir_cont_test = os.listdir(path_to_data+'testing/')
    
    if (len(dir_cont_train) != 0 and len(dir_cont_val) != 0 and
      len(dir_cont_test) != 0):
      data_avail = True
    else:
      data_avail = False
      print('\nDataset directory exists, but some data is missing!\n')
      
  else:
    data_avail = False
    path_to_data = 'datasets/{}/processed/'.format(name)
    print('\nDataset is not available on {}!\n'.format(path_to_data))
    
  # download data if not available or missing files
  if not data_avail:
    download_data.download(name, path_to_data, token)
    
  # set paths and read the directories
  path_to_train = path_to_data + 'training/'
  path_to_val = path_to_data + 'validation/'
  path_to_test = path_to_data + 'testing/'
  train_cont = os.listdir(path_to_train)
  val_cont = os.listdir(path_to_val)
  test_cont = os.listdir(path_to_test)
  
  # set empty dataframes
  train = pd.DataFrame()
  val = pd.DataFrame()
  test = pd.DataFrame()
  
  # iterate and concatenate
  print("Loading training data.")
  pbar = tqdm(total=len(train_cont))
  for f_name in train_cont:
    train = pd.concat((train, pd.read_csv(path_to_train+f_name)))
    pbar.update(1)
  print("Loading validation data.")
  pbar = tqdm(total=len(val_cont))
  for f_name in val_cont:
    val = pd.concat((val, pd.read_csv(path_to_val+f_name)))
    pbar.update(1)
  print("Loading testing data.")
  pbar = tqdm(total=len(test_cont))
  for f_name in test_cont:
    test = pd.concat((test, pd.read_csv(path_to_test+f_name)))
    pbar.update(1)
    
  # convert to numpy arrays
  train = train.to_numpy()
  val = val.to_numpy()
  test = test.to_numpy()
  
  # set and return value
  return_value = (train, val, test)
  return return_value
  
  
