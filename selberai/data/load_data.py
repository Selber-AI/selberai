import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

import selberai.data.download_data as download_data

class Dataset:
  """
  """
  
  def __init__(self, 
    train: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | tuple[pd.DataFrame, pd.DataFrame], 
    val: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | tuple[pd.DataFrame, pd.DataFrame],
    test: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | tuple[pd.DataFrame, pd.DataFrame], 
    add: dict[str, np.ndarray] | None):
    
    
    self.train = train
    self.val = val
    self.test = test
    self.add = add
    

def load(name: str, subtask: str, sample_only: bool=False, form: str='uniform', 
  path_to_data: str=None, path_to_token: str=None) -> Dataset:
  """
  Loads the data into memory and returns it.
  
  Parameters:
   - name (str): full name of the dataset to load.
   - subtask (str): subtask of the dataset to load.
   - sample_only (bool): if True, loads only 1 csv file for each split. 
   Default: False.
   - form (str): determines the output format. Can be 'uniform', 'tabular' or 
   'dataframe'. Default: 'uniform'
   - path_to_data (str):  path to the data folder containing all datasets.
   - path_to_token (str): path to the token for https://dataverse.harvard.edu

  Returns a Dataset object with the attributes 'train', 'val', 'test' and 'add'.
  
  If the 'form' parameter is 'uniform', each attribute contains a dictionary 
  with x_t, x_s, x_st, y data depending on the dataset.

  If the 'form' parameter is 'tabular', each split contains a tuple with the 
  full features and labels in np.ndarray format, while 'add' contains dict.
  
  if the 'form' parameter is 'dataframe', each attribute contains a single 
  pandas dataframe with all features and labels.
  """
  
  # set standard path to data if not provided
  if path_to_data is None:
    path_to_data = 'datasets/{}/processed/'.format(name)
    
  else:
    path_to_data += name + '/'
  
  # extend path 
  path_to_data += subtask + '/'
  
  # list directory of dataset
  dir_cont = set(os.listdir(path_to_data))
  
  # set paths and read the directories
  path_to_train = path_to_data + 'training/'
  path_to_val = path_to_data + 'validation/'
  path_to_test = path_to_data + 'testing/'
  
  # check if dataset available
  if ('training' in dir_cont and 'testing' in dir_cont and 
    'validation' in dir_cont):
    
    # read content of available train, val and test directories
    dir_cont_train = os.listdir(path_to_train)
    dir_cont_val = os.listdir(path_to_val)
    dir_cont_test = os.listdir(path_to_test)
    
    # check if content is non-zero
    if (len(dir_cont_train) != 0 and len(dir_cont_val) != 0 and
      len(dir_cont_test) != 0):
      data_avail = True
    
    else:
      data_avail = False
      print('\nDataset available, but some datasets are missing files!\n')
    
  else:
    data_avail = False
    print('\nDataset is not available on {}!\n'.format(path_to_data))
  
  # download data if not available or missing files
  if not data_avail:
    # TO DO: implement download of subtask data only
    path_to_data = path_to_data[:-(len(subtask)+1)]
    download_data.download(name, subtask, path_to_data, path_to_token)
  
  
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
  
  # iterate over train and concatenate
  print("Loading training data.")
  for f_name in tqdm(train_cont):
    train = pd.concat((train, pd.read_csv(path_to_train+f_name)))
    
  # iterate over val and concatenate
  print("Loading validation data.")
  for f_name in tqdm(val_cont):
    val = pd.concat((val, pd.read_csv(path_to_val+f_name)))
    
  # iterate over test and concatenate
  print("Loading testing data.")
  for f_name in tqdm(test_cont):
    test = pd.concat((test, pd.read_csv(path_to_test+f_name)))
    
  
  ###
  # Convert to unified data representation and potentially load additional ###
  ###
  
  # convert BuildingElectricity to unified representation
  if name == 'BuildingElectricity':

    # convert train, val test
    train = convert_be(train, form)
    val = convert_be(val, form)
    test = convert_be(test, form)

    ### Set additional ###
    # set path to additional data
    path = path_to_data+'additional/id_histo_map.csv'
    
    # read additional data
    add = {'x_s': pd.read_csv(path)}
      
  # convert Uber Movement to unified representation
  elif name == 'UberMovement':
    
    # if Uber Movement subtask is not 'cities_10', and sample_only=False,
    # then additonal data must be loaded
    train, val, test = load_correction_um(train, val, test, subtask, 
      sample_only)
    
    # convert train, val test
    train = convert_um(train, form)
    val = convert_um(val, form)
    test = convert_um(test, form)
    
    add = None
  
  
  # convert WindFarm to unified representation
  elif name == 'WindFarm':
    
    # convert train, val test
    train = convert_wf(train, form)
    val = convert_wf(val, form)
    test = convert_wf(test, form)
    
    add = None
      
      
  # convert ClimART to unified representation
  elif name == 'ClimART':
    
    # convert train, val, test
    train = convert_ca(train, subtask, form)
    val = convert_ca(val, subtask, form)
    test = convert_ca(test, subtask, form)
      
    add = None
      
  elif name == 'Polianna':
    
    # convert train, val, test
    train = convert_pa(train, subtask, form)
    val = convert_pa(val, subtask, form)
    test = convert_pa(test, subtask, form)
    
    ### Set additional ###  
    # set path to additional data
    path = path_to_data + 'additional/article_tokenized.json'
    add = {}
    
    # load article data
    with open(path, 'r') as json_file:
      add['x_st'] = json.load(json_file)
    
    # load label data
    if subtask == 'text_level':
      path = path_to_data + 'additional/annotation_labels.json'
      with open(path, 'r') as json_file:
        add['y_st'] = json.load(json_file)
        
  ### Set and return values as Dataset object ###
  dataset = Dataset(train, val, test, add)
  
  return dataset
  


def load_correction_um(train: pd.DataFrame, val: pd.DataFrame, 
  test: pd.DataFrame, subtask: str, sample_only: bool) -> (pd.DataFrame, 
  pd.DataFrame, pd.DataFrame):
  """ TODO
  """
  pass


def convert_um(df: pd.DataFrame, form: str) -> dict:
  """ TODO
  """
  pass


def convert_pa(df: pd.DataFrame, subtask: str, form: str) -> (
  dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] 
  | tuple[pd.DataFrame, pd.DataFrame]):
  """
  """
  # set starting and end indices of tabular features
  end_t = 3
  end_s = end_t + 2
  end_st = end_s + 1
  

  if form == 'uniform':
    data_dict = {}
    data_dict['x_t'] = df.iloc[:, :end_t].to_numpy()
    data_dict['x_s'] = df.iloc[:, end_t:end_s].to_numpy()
    data_dict['x_st'] = df.iloc[:, end_s:end_st].to_numpy()
    if subtask == 'article_level':
      data_dict['y_st'] = df.iloc[:, end_st:].to_numpy()
    elif subtask == 'text_level':
      data_dict['y_st'] = data_dict['x_st']
    
    return data_dict
  
  elif form == 'tabular':
    features = df.iloc[:, :end_st].to_numpy()
    labels = df.iloc[:, end_st:].to_numpy()

    return features, labels
    
  elif form == 'dataframe':
    features = df.iloc[:, :end_st]
    labels = df.iloc[:, end_st:]

    return features, labels
  
  
def convert_ca(df: pd.DataFrame, subtask: str, form: str) -> (
  dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray] | 
  tuple[pd.DataFrame, pd.DataFrame]):
  """
  """
  # set values from config file
  n_data = len(df.index)
  n_layers = 49
  n_levels = 50
  vars_global = 79 # 82 globals. 3 coordinates x,y,z extracted. are in x_s
  vars_levels = 4
  
  if subtask == 'pristine':
    vars_layers = 14
    
  elif subtask == 'clear_sky':
    vars_layers = 45
  
  # set starting and end indices of tabular features
  end_t = 2
  end_s = end_t + 3
  end_st_1 = end_s + vars_global
  end_st_2 = end_st_1 + vars_layers * n_layers
  end_st_3 = end_st_2 + vars_levels * n_levels
  end_y1 = end_st_3 + 2 * n_layers
  
  if form == 'uniform':
    data_dict = {}
    data_dict['x_t'] = df.iloc[:, :end_t].to_numpy()
    data_dict['x_s'] = df.iloc[:, end_t:end_s].to_numpy()
    data_dict['x_st_1'] = df.iloc[:, end_s:end_st_1].to_numpy()
    data_dict['x_st_2'] = df.iloc[:, end_st_1:end_st_2].to_numpy()
    data_dict['x_st_3'] = df.iloc[:, end_st_2:end_st_3].to_numpy()
    data_dict['y_st_1'] = df.iloc[:, end_st_3:end_y1].to_numpy()
    data_dict['y_st_2'] = df.iloc[:, end_y1:].to_numpy()
    
    # reshape arrays
    data_dict['x_st_2'] = np.reshape(data_dict['x_st_2'], 
      (n_data, n_layers, vars_layers), order='C')
    data_dict['x_st_3'] = np.reshape(data_dict['x_st_3'], 
      (n_data, n_levels, vars_levels), order='C')
    data_dict['y_st_1'] = np.reshape(data_dict['y_st_1'], 
      (n_data, n_layers, 2), order='C')
    data_dict['y_st_2'] = np.reshape(data_dict['y_st_2'], 
      (n_data, n_levels, 4), order='C')
    
    return data_dict
    
  elif form == 'tabular':
    features = df.iloc[:, :end_st_3].to_numpy()
    labels = df.iloc[:, end_st_3:].to_numpy()

    return features, labels
    
  elif form == 'dataframe':
    features = df.iloc[:, :end_st_3]
    labels = df.iloc[:, end_st_3:]

    return features, labels
  
  
def convert_wf(df: pd.DataFrame, form: str) -> (dict[str, np.ndarray] | 
  tuple[np.ndarray, np.ndarray] | tuple[pd.DataFrame, pd.DataFrame]):
  """
  """
  # set values from config file
  hist_window = 288
  pred_window = 288
  n_times = 3
  n_states = 10
  n_data = len(df.index)
  
  # set starting and end indices of tabular features
  end_s = 2
  end_t1 = end_s + hist_window * n_times
  end_st = end_t1 + hist_window * n_states
  end_t2 = end_st + pred_window * n_times
  
  if form == 'uniform':
    data_dict = {}
    data_dict['x_s'] = df.iloc[:, :end_s].to_numpy()
    data_dict['x_t_1'] = df.iloc[:, end_s:end_t1].to_numpy().astype(int)
    data_dict['x_st'] = df.iloc[:, end_t1:end_st].to_numpy()
    data_dict['x_t_2'] = df.iloc[:, end_st:end_t2].to_numpy().astype(int)
    data_dict['y_st'] = df.iloc[:, end_t2:].to_numpy()
    
    # either order='C' with shape (n_data, n_states, hist_window)
    # or order='F' with shape (n_data, hist_window, n_states)
    data_dict['x_t_1'] = np.reshape(data_dict['x_t_1'], 
      (n_data, n_times, hist_window), order='C')
    data_dict['x_t_2'] = np.reshape(data_dict['x_t_2'], 
      (n_data, n_times, pred_window), order='C')
    data_dict['x_st'] = np.reshape(data_dict['x_st'], 
      (n_data, n_states, hist_window), order='C')
      
    return data_dict
    
  elif form == 'tabular':
    features = df.iloc[:, :end_t2].to_numpy()
    labels = df.iloc[:, end_t2:].to_numpy()

    return features, labels
    
  elif form == 'dataframe':
    features = df.iloc[:, :end_t2]
    labels = df.iloc[:, end_t2:]

    return features, labels
  
  
def convert_be(df: pd.DataFrame, form: str) -> (dict[str, np.ndarray] | 
  tuple[np.ndarray, np.ndarray] | tuple[pd.DataFrame, pd.DataFrame]):
  """
  """
  
  # set values from config file
  hist_window = 24
  n_states = 9
  n_data = len(df.index)
  
  # set starting and end indices of tabular features
  end_t = 5
  start_st = end_t + 1
  end_st = start_st + hist_window * n_states
  
  if form == 'uniform':
    data_dict = {}
    data_dict['x_t'] = df.iloc[:, :end_t].to_numpy().astype(int)
    data_dict['x_s'] = df.iloc[:, end_t].to_numpy().astype(int)
    data_dict['x_st'] = df.iloc[:, start_st:end_st].to_numpy()
    data_dict['y_st'] = df.iloc[:, end_st:].to_numpy()
    
    # either order='C' with shape (n_data, n_states, hist_window)
    # or order='F' with shape (n_data, hist_window, n_states)
    data_dict['x_st'] = np.reshape(data_dict['x_st'], 
      (n_data, n_states, hist_window), order='C')
    
    return data_dict
    
  elif form == 'tabular':
    features = df.iloc[:, :end_st].to_numpy()
    labels = df.iloc[:, end_st:].to_numpy()

    return features, labels
    
  elif form == 'dataframe':
    features = df.iloc[:, :end_st]
    labels = df.iloc[:, end_st:]
    
    return features, labels
    
  
