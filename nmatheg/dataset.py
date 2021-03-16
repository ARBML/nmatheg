
import tensorflow as tf
import tnkeeh as tn 
from datasets import load_dataset
import os 
from .utils import get_preprocessing_args

def write_split(dataset, config, data_config, split = 'train'):
    data = []
    lbls = []
    
    dataset_name = config['dataset']['dataset_name']
    text = data_config[dataset_name]['text']
    label = data_config[dataset_name]['label']

    args = get_preprocessing_args(config)
    cleaner = tn.Tnqeeh(**args)
    dataset = cleaner.clean_hf_dataset(dataset, text)

    for sample in dataset:
        data.append(sample[text])
        lbls.append(str(sample[label]))
    
    open(f'data/{split}_data.txt', 'w').write(('\n').join(data))
    open(f'data/{split}_lbls.txt', 'w').write(('\n').join(lbls))

def split_dataset(dataset):
    if not('test' in dataset):
        train_valid_dataset = dataset['train'].train_test_split(test_size=0.1)
        train_valid_dataset['valid'] = train_valid_dataset.pop('test')

        train_test_dataset = train_valid_dataset['train'].train_test_split(test_size=0.1)
        
        dataset['train'] = train_test_dataset['train']
        dataset['test'] = train_test_dataset['test']
        dataset['valid'] = train_valid_dataset['valid']

    elif not('valid' in dataset):
        train_valid_dataset = dataset['train'].train_test_split(test_size=0.1)
        train_valid_dataset['valid'] = train_valid_dataset.pop('test')
        dataset['train'] = train_valid_dataset['train']
        dataset['valid'] = train_valid_dataset['valid']
        dataset['test'] = dataset['test']
    return dataset 

def write_dataset(dataset_name, config, data_config):
    os.makedirs('data', exist_ok=True)
    dataset = load_dataset(dataset_name)
    dataset = split_dataset(dataset)
    for split in dataset:
        write_split(dataset[split], config, data_config, split = split)
    
def prepare_data(dataset_name, config, data_config):
  write_dataset(dataset_name, config, data_config)

  train_text = open('data/train_data.txt', 'r').read().splitlines()
  train_lbls = [int(lbl) for lbl in open('data/train_lbls.txt', 'r').read().splitlines()]
  valid_text = open('data/valid_data.txt', 'r').read().splitlines()
  valid_lbls = [int(lbl) for lbl in open('data/valid_lbls.txt', 'r').read().splitlines()]
  test_text = open('data/test_data.txt', 'r').read().splitlines()
  test_lbls = [int(lbl) for lbl in open('data/test_lbls.txt', 'r').read().splitlines()]
  assert len(train_text) == len(train_lbls)
  assert len(test_text) == len(test_lbls)
  assert len(valid_text) == len(valid_lbls)
  return train_text, valid_text, test_text, train_lbls, valid_lbls, test_lbls

def tokenize_data(dataset_name, config, data_config, tokenizer, vocab_size = 10000, max_tokens = 128):
  train_text, valid_text, test_text, train_lbls, valid_lbls, test_lbls = prepare_data(dataset_name, config, data_config)
  tokenizer = tokenizer(vocab_size = vocab_size)
  tokenizer.train('data/train_data.txt')
  train_data = tokenizer.encode_sentences(train_text, out_length=max_tokens)
  valid_data = tokenizer.encode_sentences(valid_text, out_length=max_tokens)
  test_data = tokenizer.encode_sentences(test_text, out_length=max_tokens)
  return (train_data, train_lbls), (valid_data, valid_lbls), (test_data, test_lbls)

def create_dataset(data, batch_size = 256, buffer_size = 50000):
  train_data, valid_data, test_data = data
  train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

  valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
  valid_dataset = valid_dataset.batch(batch_size)

  test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
  test_dataset = test_dataset.batch(batch_size)
  return train_dataset, valid_dataset, test_dataset
