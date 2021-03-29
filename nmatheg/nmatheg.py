
import os 
from .dataset import create_dataset_simple, create_dataset_bert
from .models import SimpleClassificationModel, BERTClassificationModel
from .utils import get_tokenizer
import configparser

class TrainStrategy:
  def __init__(self, config_path):

    config = configparser.ConfigParser()
    config.read(config_path)

    data_config = configparser.ConfigParser()
    rel_path = os.path.dirname(__file__)
    data_ini_path = os.path.join(rel_path, "datasets.ini")
    data_config.read(data_ini_path)

    vocab_size = int(config['tokenization']['vocab_size'])

    dataset_name = config['dataset']['dataset_name']
    num_labels = int(data_config[dataset_name]['num_labels'])

    batch_size = int(config['train']['batch_size'])
    self.epochs = int(config['train']['epochs'])
    model_name =  config['model']['model_name']

    self.print_every = int(config['log']['print_every'])

    if 'bert' in model_name:
      self.datasets = create_dataset_bert(dataset_name, config, 
      data_config, batch_size = batch_size)
      self.model = BERTClassificationModel(model_name, num_labels)
    else:
      self.datasets = create_dataset_simple(dataset_name, config, 
      data_config, batch_size = batch_size)
      self.model = SimpleClassificationModel(vocab_size, num_labels)

  def start(self):
    self.model.train(self.datasets, epochs = self.epochs,
              print_every = self.print_every)