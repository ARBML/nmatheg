
import os 
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GRU, Embedding, Dense, Input, Dropout, Bidirectional
import numpy as np
import matplotlib.pyplot as plt
import logging 
import sys
from .dataset import tokenize_data, create_dataset
from .models import ClassificationModel
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

    max_tokens = int(config['tokenization']['max_tokens'])
    tokenizer_name = config['tokenization']['tokenizer_name']
    tokenizer = get_tokenizer(tokenizer_name)
    vocab_size = int(config['tokenization']['vocab_size'])

    dataset_name = config['dataset']['dataset_name']
    num_classes = int(data_config[dataset_name]['num_classes'])

    batch_size = int(config['train']['batch_size'])
    train_dir = config['train']['dir']
    self.epochs = int(config['train']['epochs'])

    data = tokenize_data(dataset_name, config, data_config, tokenizer, 
    vocab_size = vocab_size, max_tokens=max_tokens)
    self.datasets = create_dataset(data, batch_size = batch_size)
    ckpt_dir = f'{train_dir}/ckpts/vocab_size_{vocab_size}/{tokenizer_name}/'
    self.model = ClassificationModel(vocab_size, num_classes, ckpt_dir)

  def start(self):
    self.model.train(self.datasets, epochs = self.epochs)