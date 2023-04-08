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
    
  ###
  # Load training, validation and testing ###
  ###
  
  # set paths and read the directories
  path_to_train = path_to_data + 'training/'
  path_to_val = path_to_data + 'validation/'
  path_to_test = path_to_data + 'testing/'
  
  # read directory content
  train_cont = os.listdir(path_to_train)
  val_cont = os.listdir(path_to_val)
  test_cont = os.listdir(path_to_test)
  
  # reduce directory content to single file only if sample_only==True
  if sample_only:
    train_cont = train_cont[:1]
    val_cont = val_cont[:1]
    test_cont = test_cont[:1]
  
  # set empty dataframes
  train = pd.DataFrame()
  val = pd.DataFrame()
  test = pd.DataFrame()
  
  # set additional default to None. Replace for respective datasets
  add = None
  
  # iterate over train, val, test data files and concatenate
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
  
 
  ###
  # Convert to unified data representation and potentially load additional ###
  ###
  
  if name == 'BuildingElectricity':
    train = convert_be(config[name], train)
    val = conver_be(config[name], val)
    test = convert_be(config[name], test)
    add = load_add_be()
  elif name == 'WindFarm':
    print('To Do: Needs to be implemented!')

  
  # set and return value
  return_value = (train, val, test, add)
  return return_value
  

def convert_be(config_be: dict, dataset: pd.DataFrame) -> np.array:
  """
  """
  
  dataset = dataset.to_numpy()
  
  return dataset
