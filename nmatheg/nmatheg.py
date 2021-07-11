
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
    
    self.train_config, self.model_config = create_configs(self.config, self.data_config)
    self.datasets = create_dataset(self.config, self.data_config)
    model_names = self.model_config['model_name'].split(',')

    for model_name in model_names:
      self.model_config['model_name'] = model_name
      if 'bert' in model_name:
        self.model = BERTClassificationModel(self.model_config)
      else:
        self.model = SimpleClassificationModel(self.model_config)

      self.model.train(self.datasets, **self.train_config)