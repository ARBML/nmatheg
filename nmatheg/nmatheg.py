
import os 
from .dataset import create_dataset
from .models import SimpleClassificationModel, BERTClassificationModel
from .utils import get_tokenizer
from .configs import create_configs
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

    for dataset_name in dataset_names:
      self.self.config['dataset']['dataset_name'] = dataset_name

      for model_name in model_names:
        self.config['model']['model_name'] = model_name
        self.train_config, self.model_config = create_configs(self.config, self.data_config)
        self.datasets = create_dataset(self.config, self.data_config)
        
        if 'bert' in model_name:
          self.model = BERTClassificationModel(self.model_config)
        else:
          self.model = SimpleClassificationModel(self.model_config)

        results = self.model.train(self.datasets, **self.train_config) 
        results['model_name'] = model_name
        results['dataset_name'] = dataset_name 
        self.model.wipe_memory()     