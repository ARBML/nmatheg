
import tnkeeh as tn 
from datasets import load_dataset
import os 
from .utils import get_preprocessing_args
from transformers import AutoTokenizer
import torch
from .utils import get_tokenizer
from .preprocess_ner import aggregate_tokens, tokenize_and_align_labels
from .preprocess_qa import prepare_features, prepare_validation_features
import copy 

def split_dataset(dataset, seed = 42):
    if not('test' in dataset):
        train_valid_dataset = dataset['train'].train_test_split(test_size=0.1, seed = seed)
        train_valid_dataset['valid'] = train_valid_dataset.pop('test')

        train_test_dataset = train_valid_dataset['train'].train_test_split(test_size=0.1, seed = seed)
        
        dataset['train'] = train_test_dataset['train']
        dataset['test'] = train_test_dataset['test']
        dataset['valid'] = train_valid_dataset['valid']

    elif not('valid' in dataset):
        train_valid_dataset = dataset['train'].train_test_split(test_size=0.1, seed = seed)
        train_valid_dataset['valid'] = train_valid_dataset.pop('test')
        dataset['train'] = train_valid_dataset['train']
        dataset['valid'] = train_valid_dataset['valid']
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
    task_name = config['dataset']['task']

    # clean and load data
    dataset = load_dataset(dataset_name)
    if task_name != 'question_answering':
        dataset = clean_dataset(dataset, config, data_config)

    examples = copy.deepcopy(dataset)
    if task_name == 'text_classification':
        # tokenize data
        if 'bert' in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, model_max_length = 512)
            dataset = dataset.map(lambda examples:tokenizer(examples[data_config[dataset_name]['text']], truncation=True, padding='max_length'), batched=True)
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        else:
            write_data_for_train(dataset['train'], data_config[dataset_name]['text'])
            tokenizer = get_tokenizer(tokenizer_name)
            tokenizer = tokenizer(vocab_size = vocab_size)
            tokenizer.train('data.txt')
            dataset = dataset.map(lambda examples:{'input_ids': tokenizer.encode_sentences(examples[data_config[dataset_name]['text']], out_length= max_tokens)}, batched=True)
            columns=['input_ids', 'labels'] 
        
        dataset = dataset.map(lambda examples:{'labels': examples[data_config[dataset_name]['label']]}, batched=True)

    elif task_name == 'token_classification':
        dataset = aggregate_tokens(dataset, config, data_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        for split in dataset:
            dataset[split] = dataset[split].map(lambda x: tokenize_and_align_labels(x, tokenizer, config, data_config)
                                                , batched=True, remove_columns=dataset[split].column_names)

        columns=['input_ids', 'attention_mask', 'labels']
    
    elif task_name == 'question_answering':
        #TODO fix these 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        dataset['train'] = dataset['train'].map(lambda x: prepare_features(x, tokenizer)
                                                , batched=True, remove_columns=dataset['train'].column_names)

        dataset['validation'] = dataset['validation'].map(lambda x: prepare_features(x, tokenizer)
                                                , batched=True, remove_columns=dataset['validation'].column_names)

        columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions']
        

    #create loaders

    if task_name == 'question_answering':
        # #TODO fix these 
        # for split in dataset:
        #     if split == 'train':
        #         dataset[split].set_format(type='torch', columns = columns)
        #         dataset[split] = torch.utils.data.DataLoader(dataset[split], batch_size=batch_size)
        pass 
    else: 
        dataset = split_dataset(dataset)
        examples = split_dataset(examples)
        for split in dataset:
            dataset[split].set_format(type='torch', columns=columns)
            dataset[split] = torch.utils.data.DataLoader(dataset[split], batch_size=batch_size)
    
    if task_name == 'question_answering':
        return [dataset['train'], dataset['validation'], dataset['validation']], [examples['train'], examples['validation'], examples['validation']]
    return [dataset['train'], dataset['valid'], dataset['test']]
