import os
from tqdm import tqdm
import pandas as pd
import numpy as np

import selberai.data.download_data as download_data

class Dataset:
  """
  """
  
  def __init__(self, train: np.array, val: np.array, test: np.array, 
    add: pd.DataFrame):
    
    self.train = train
    self.val = val
    self.test = test
    self.add = add
    

def load(name: str, sample_only=False, path_to_data=None, path_to_token=None
  ) -> Dataset:
  """
  """
  
  # set standard path to data if not provided
  if path_to_data is None:
    path_to_data = 'datasets/{}/processed/'.format(name)
  
  # list directory of dataset
  dir_cont = set(os.listdir(path_to_data))
  
  # set paths and read the directories
  path_to_train = path_to_data + 'training/'
  path_to_val = path_to_data + 'validation/'
  path_to_test = path_to_data + 'testing/'
  
  # check if dataset available
  if ('training' in dir_cont and 'testing' in dir_cont and 
    'validation' in dir_cont):
    dir_cont_train = os.listdir(path_to_train)
    dir_cont_val = os.listdir(path_to_val)
    dir_cont_test = os.listdir(path_to_test)
    
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
    download_data.download(name, path_to_data, path_to_token)
    
    
  ###
  # Load training, validation and testing ###
  ###
  
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
  
  # convert BuildingElectricity to unified representation
  if name == 'BuildingElectricity':
    path = path_to_data + 'additional/building_images_pixel_histograms_rgb.csv'
    add = {'id_histo_map': pd.read_csv(path)}
    train, val, test = convert_be(train), convert_be(val), convert_be(test)
  
  # convert WindFarm to unified representation
  elif name == 'WindFarm':
    train, val, test = convert_wf(train), convert_wf(val), convert_wf(test)

  # set and return values as Dataset object
  dataset = Dataset(train, val, test, add)
  return dataset
  
  
def convert_wf(dataframe: pd.DataFrame) -> dict:
  """
  """
  
  data_dict = {}
  data_dict['x_s'] = dataframe.iloc[:, :2].to_numpy()
  data_dict['x_t'] = dataframe.iloc[:, 2:5].to_numpy()
  data_dict['x_st'] = dataframe.iloc[:, 5:(5+288*10)].to_numpy()
  data_dict['y'] = dataframe.iloc[:, (5+288*10):].to_numpy()
  
  # alternative is order='C' with shape (len(data_dict['x_st']), 9, 24)
  data_dict['x_st'] = np.reshape(data_dict['x_st'], 
    (len(data_dict['x_st']), 288, 10), order='F')
  
  
  return data_dict
  
def convert_be(dataframe: pd.DataFrame) -> dict:
  """
  """
  
  data_dict = {}
  data_dict['x_t'] = dataframe.iloc[:, :5].to_numpy()
  data_dict['x_s'] = dataframe.iloc[:, 5].to_numpy()
  data_dict['x_st'] = dataframe.iloc[:, 6:(6+24*9)].to_numpy()
  data_dict['y'] = dataframe.iloc[:, (6+24*9):].to_numpy()
  
  # alternative is order='C' with shape (len(data_dict['x_st']), 9, 24)
  data_dict['x_st'] = np.reshape(data_dict['x_st'], 
    (len(data_dict['x_st']), 24, 9), order='F')
  
  
  return data_dict
  
  
