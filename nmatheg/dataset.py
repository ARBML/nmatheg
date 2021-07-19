
import tnkeeh as tn 
from datasets import load_dataset
import os 
from .utils import get_preprocessing_args
from transformers import AutoTokenizer
import torch
from .utils import get_tokenizer
from .preprocess_ner import aggregate_tokens, tokenize_and_align_labels
from .preprocess_qa import prepare_features
import copy 

def split_dataset(dataset, config, data_config, seed = 42):
    dataset_name = config['dataset']['dataset_name']
    split_names = ['train', 'valid', 'test']

    for i, split_name in enumerate(['train', 'valid', 'test']):
        if split_name in data_config[dataset_name]:
            split_names[i] = data_config[dataset_name][split_name]
            dataset[split_name] = dataset[split_names[i]]

    #create validation split
    if 'valid' not in dataset:
        train_valid_dataset = dataset['train'].train_test_split(test_size=0.1, seed = seed)
        dataset['valid'] = train_valid_dataset.pop('test')
        dataset['train'] = train_valid_dataset['train']

    #create training split 
    if 'test' not in dataset:
        train_valid_dataset = dataset['train'].train_test_split(test_size=0.1, seed = seed)
        dataset['test'] = train_valid_dataset.pop('test')
        dataset['train'] = train_valid_dataset['train']  
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
    task_name = data_config[dataset_name]['task']

    # clean and load data
    dataset = load_dataset(dataset_name)
    if task_name != 'qa':
        dataset = clean_dataset(dataset, config, data_config)

    dataset = split_dataset(dataset, config, data_config)
    examples = copy.deepcopy(dataset)

    if task_name == 'cls':
        # tokenize data
        # if 'bert' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, model_max_length = 512)
        dataset = dataset.map(lambda examples:tokenizer(examples[data_config[dataset_name]['text']], truncation=True, padding='max_length'), batched=True)
        columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        # else:
        #     write_data_for_train(dataset['train'], data_config[dataset_name]['text'])
        #     tokenizer = get_tokenizer(tokenizer_name)
        #     tokenizer = tokenizer(vocab_size = vocab_size)
        #     tokenizer.train('data.txt')
        #     dataset = dataset.map(lambda examples:{'input_ids': tokenizer.encode_sentences(examples[data_config[dataset_name]['text']], out_length= max_tokens)}, batched=True)
        #     columns=['input_ids', 'labels'] 
        
        dataset = dataset.map(lambda examples:{'labels': examples[data_config[dataset_name]['label']]}, batched=True)

    elif task_name == 'ner':
        dataset = aggregate_tokens(dataset, config, data_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        for split in dataset:
            dataset[split] = dataset[split].map(lambda x: tokenize_and_align_labels(x, tokenizer, config, data_config)
                                                , batched=True, remove_columns=dataset[split].column_names)

        columns=['input_ids', 'attention_mask', 'labels']
    
    elif task_name == 'qa':
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        for split in dataset:
          dataset[split] = dataset[split].map(lambda x: prepare_features(x, tokenizer)
                                                , batched=True, remove_columns=dataset[split].column_names)
   
    #create loaders 
    if task_name != 'qa': 
        for split in dataset:
            dataset[split].set_format(type='torch', columns=columns)
            dataset[split] = torch.utils.data.DataLoader(dataset[split], batch_size=batch_size)
    
    return [dataset['train'], dataset['valid'], dataset['test']], [examples['train'], examples['valid'], examples['test']]
