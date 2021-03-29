
import tnkeeh as tn 
from datasets import load_dataset
import os 
from .utils import get_preprocessing_args
from transformers import AutoTokenizer
import torch
from .utils import get_tokenizer


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



def clean_dataset(dataset, config, data_config):
    dataset_name = config['dataset']['dataset_name']
    text = data_config[dataset_name]['text']

    args = get_preprocessing_args(config)
    cleaner = tn.Tnkeeh(**args)
    dataset = cleaner.clean_hf_dataset(dataset, text)
    return dataset 

def write_data_for_train(dataset, text):
    data = []
    for sample in dataset:
        data.append(sample[text])    
    open(f'data.txt', 'w').write(('\n').join(data))

def create_dataset_simple(dataset_name, config, data_config, batch_size = 32):
    tokenizer_name = config['tokenization']['tokenizer_name']
    max_tokens = int(config['tokenization']['max_tokens'])
    tokenizer = get_tokenizer(tokenizer_name)
    text = data_config[dataset_name]['text']

    def encode(example):
        example['text'] = tokenizer.encode_sentences(example['text'], out_length= max_tokens)
        return example

    dataset = load_dataset(dataset_name)
    dataset = clean_dataset(dataset, config, data_config)
    write_data_for_train(dataset['train'], text)

    vocab_size  = int(config['tokenization']['vocab_size'])
    tokenizer = tokenizer(vocab_size = vocab_size)
    tokenizer.train('data.txt')

    dataset = dataset.map(encode, batched=True)
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    splits = split_dataset(dataset)
    for split in splits:
        dataset[split].set_format(type='torch', columns=['text', 'labels'])
        dataset[split] = torch.utils.data.DataLoader(dataset[split], batch_size=batch_size)
    return [dataset['train'], dataset['valid'], dataset['test']]

def create_dataset_bert(dataset_name, config, data_config, batch_size = 32):
    def encode(examples):
      return tokenizer(examples['text'], truncation=True, padding='max_length')


    model_name =  config['model']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    dataset = load_dataset(dataset_name)
    dataset = clean_dataset(dataset, config, data_config)
    dataset = dataset.map(encode, batched=True)
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    splits = split_dataset(dataset)

    for split in splits:
        dataset[split].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        dataset[split] = torch.utils.data.DataLoader(dataset[split], batch_size=batch_size)
    return [dataset['train'], dataset['valid'], dataset['test']]

