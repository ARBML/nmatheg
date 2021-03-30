
import os 
from .dataset import create_dataset
from .models import SimpleClassificationModel, BERTClassificationModel
from .utils import get_tokenizer
from .configs import create_configs
import configparser

class TrainStrategy:
  def __init__(self, config_path):

    config = configparser.ConfigParser()
    config.read(config_path)

    data_config = configparser.ConfigParser()
    rel_path = os.path.dirname(__file__)
    data_ini_path = os.path.join(rel_path, "datasets.ini")
    data_config.read(data_ini_path)

    self.train_config, self.model_config = create_configs(config, data_config)
    self.datasets = create_dataset(config, data_config)

    if 'bert' in self.model_config['model_name']:
      self.model = BERTClassificationModel(self.model_config)
    else:
      self.model = SimpleClassificationModel(self.model_config)

  def start(self):
    self.model.train(self.datasets, **self.train_config)