
import os 
from .dataset import create_dataset
from .models import SimpleClassificationModel, BERTTextClassificationModel,BERTTokenClassificationModel,BERTQuestionAnsweringModel
from .configs import create_configs, create_default_config
import pandas as pd
import configparser

class TrainStrategy:
  def __init__(self, config_path = None, datasets = '', models = ''):
    if config_path == None:
      self.config = create_default_config()
      self.config['dataset'] = {'dataset_name' : datasets}
      self.config['model'] = {'model_name' : models}
    else:
      self.config = configparser.ConfigParser()
      self.config.read(config_path)

    self.data_config = configparser.ConfigParser()
    rel_path = os.path.dirname(__file__)
    data_ini_path = os.path.join(rel_path, "datasets.ini")
    self.data_config.read(data_ini_path)

  def start(self):
    model_names = self.config['model']['model_name'].split(',')
    dataset_names = self.config['dataset']['dataset_name'].split(',')

    output = [] 
    for dataset_name in dataset_names:
      self.config['dataset']['dataset_name'] = dataset_name
      task_name = self.data_config[dataset_name]['task']

      for model_name in model_names:
        self.config['model']['model_name'] = model_name
        self.train_config, self.model_config = create_configs(self.config, self.data_config)
        self.datasets, self.examples = create_dataset(self.config, self.data_config)
        
        if task_name == 'cls':
          if 'bert' in model_name:
            self.model = BERTTextClassificationModel(self.model_config)
          else:
            self.model = SimpleClassificationModel(self.model_config)

        elif task_name == 'ner':
          self.model = BERTTokenClassificationModel(self.model_config)

        elif task_name == 'qa':
          self.model = BERTQuestionAnsweringModel(self.model_config)

        if task_name == 'qa':
          results = self.model.train(self.datasets, self.examples, **self.train_config) 
        else:
          results = self.model.train(self.datasets, **self.train_config) 

        results['model_name'] = model_name
        results['dataset_name'] = dataset_name 
        output.append(results)
        self.model.wipe_memory()
    
    model_results = {model_name:[0]*len(dataset_names) for model_name in model_names}
    for row in output:
      dataset_name = row['dataset_name']
      task_name = self.data_config[dataset_name]['task']
      if task_name == 'qa':
        metric = 'f1'
      else:
        metric = 'accuracy' 
      
      model_results[row['model_name']][dataset_names.index(dataset_name)] = round(row[metric]*100, 2)

    rows = []
    for model_name in model_results:
      rows.append([model_name]+model_results[model_name])

    return pd.DataFrame(rows, columns = ['Model']+dataset_names)     