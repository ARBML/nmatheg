
import os 
from .dataset import create_dataset
from .models import SimpleClassificationModel, BERTTextClassificationModel,BERTTokenClassificationModel,BERTQuestionAnsweringModel
from .configs import create_configs, create_default_config
import pandas as pd
import configparser

class TrainStrategy:
  def __init__(self, config_path = None, datasets = '', models = '', **kwargs):
    if config_path == None:
      self.config = create_default_config(**kwargs)
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
    model_names = [m.strip() for m in self.config['model']['model_name'].split(',')]
    dataset_names = [d.strip() for d in self.config['dataset']['dataset_name'].split(',')]

    output = []
    dataset_metrics = [] 
    for dataset_name in dataset_names:
      self.config['dataset']['dataset_name'] = dataset_name
      task_name = self.data_config[dataset_name]['task']

      for model_name in model_names:
        self.config['model']['model_name'] = model_name
        self.train_config, self.model_config = create_configs(self.config, self.data_config)
        self.datasets, self.examples = create_dataset(self.config, self.data_config)
        
        if task_name == 'cls':
          self.model = BERTTextClassificationModel(self.model_config)

        elif task_name == 'ner':
          self.model = BERTTokenClassificationModel(self.model_config)

        elif task_name == 'qa':
          self.model = BERTQuestionAnsweringModel(self.model_config)

        results = self.model.train(self.datasets, self.examples, **self.train_config) 

        for metric_name in results:
          if model_name == model_names[0]:
            dataset_metrics.append(dataset_name+metric_name)
        output.append([model_name, dataset_name, results])
        self.model.wipe_memory()
    
    model_results = {model_name:[0]*len(dataset_metrics) for model_name in model_names}
    metric_names = ['Model']
    dataset_names = ['']
    for row in output:
      model_name = row[0]
      dataset_name = row[1]
      metrics = row[2]
      task_name = self.data_config[dataset_name]['task']
      for metric_name in row[2]: 
        model_results[model_name][dataset_metrics.index(dataset_name+metric_name)] = round(metrics[metric_name]*100, 2)
        
        # this is only used once 
        if model_name == model_names[0]:
          metric_names.append(metric_name) 
          dataset_names.append(dataset_name) 
    
    rows = []
    for model_name in model_results:
      rows.append([model_name]+model_results[model_name])

    return pd.DataFrame(rows, columns = [dataset_names, metric_names])     