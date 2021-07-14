
import os 
from .dataset import create_dataset
from .models import SimpleClassificationModel, BERTTextClassificationModel,BERTTokenClassificationModel,BERTQuestionAnsweringModel
from .configs import create_configs
import pandas as pd
import configparser

class TrainStrategy:
  def __init__(self, config_path):

    self.config = configparser.ConfigParser()
    self.config.read(config_path)

    self.data_config = configparser.ConfigParser()
    rel_path = os.path.dirname(__file__)
    data_ini_path = os.path.join(rel_path, "datasets.ini")
    self.data_config.read(data_ini_path)

  def start(self):
    model_names = self.config['model']['model_name'].split(',')
    dataset_names = self.config['dataset']['dataset_name'].split(',')
    task_name = self.config['dataset']['task']

    output = [] 
    for dataset_name in dataset_names:
      self.config['dataset']['dataset_name'] = dataset_name

      for model_name in model_names:
        self.config['model']['model_name'] = model_name
        self.train_config, self.model_config = create_configs(self.config, self.data_config)
        self.datasets = create_dataset(self.config, self.data_config)
        
        if task_name == 'text_classification':
          if 'bert' in model_name:
            self.model = BERTTextClassificationModel(self.model_config)
          else:
            self.model = SimpleClassificationModel(self.model_config)

        elif task_name == 'token_classification':
          self.model = BERTTokenClassificationModel(self.model_config)

        elif task_name == 'question_answering':
          self.model = BERTQuestionAnsweringModel(self.model_config)

        results = self.model.train(self.datasets, **self.train_config) 
        results['model_name'] = model_name
        results['dataset_name'] = dataset_name 
        output.append(results)
        self.model.wipe_memory()
    
    model_results = {model_name:[0]*len(dataset_names) for model_name in model_names}
    for row in output:
      dataset_name = row['dataset_name']
      model_results[row['model_name']][dataset_names.index(dataset_name)] = round(row['accuracy']*100, 2)

    rows = []
    for model_name in model_results:
      rows.append([model_name]+model_results[model_name])

    return pd.DataFrame(rows, columns = ['Model']+dataset_names)     