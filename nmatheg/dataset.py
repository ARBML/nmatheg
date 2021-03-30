
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

def create_dataset(config, data_config):

    tokenizer_name = config['tokenization']['tokenizer_name']
    max_tokens = int(config['tokenization']['max_tokens'])
    vocab_size  = int(config['tokenization']['vocab_size'])

    batch_size = int(config['train']['batch_size'])
    dataset_name = config['dataset']['dataset_name']
    model_name = config['model']['model_name']

    # clean and load data
    dataset = load_dataset(dataset_name)
    dataset = clean_dataset(dataset, config, data_config)

    # tokenize data
    if 'bert' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        dataset = dataset.map(lambda examples:tokenizer(examples['text'], truncation=True, padding='max_length'), batched=True)
        columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    else:
        write_data_for_train(dataset['train'], data_config[dataset_name]['text'])
        tokenizer = get_tokenizer(tokenizer_name)
        tokenizer = tokenizer(vocab_size = vocab_size)
        tokenizer.train('data.txt')
        dataset = dataset.map(lambda examples:{'input_ids': tokenizer.encode_sentences(examples['text'], out_length= max_tokens)}, batched=True)
        columns=['input_ids', 'labels']        
    
    # split datasets 
    dataset = dataset.map(lambda examples:{'labels': examples['label']}, batched=True)
    splits = split_dataset(dataset)
    
    #create loaders 
    for split in splits:
        dataset[split].set_format(type='torch', columns=columns)
        dataset[split] = torch.utils.data.DataLoader(dataset[split], batch_size=batch_size)
    return [dataset['train'], dataset['valid'], dataset['test']]
