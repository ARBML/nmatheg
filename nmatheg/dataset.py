
import tnkeeh as tn 
from datasets import load_dataset, load_from_disk
from bpe_surgery import bpe

import os 
from .utils import get_preprocessing_args
from transformers import AutoTokenizer
import torch
from .preprocess_ner import aggregate_tokens, tokenize_and_align_labels
from .preprocess_qa import prepare_features
import copy 

def split_dataset(dataset, data_config, seed = 42):
    split_names = ['train', 'valid', 'test']

    for i, split_name in enumerate(['train', 'valid', 'test']):
        if split_name in data_config:
            split_names[i] = data_config[split_name]
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
    text = data_config['text']

    args = get_preprocessing_args(config)
    cleaner = tn.Tnkeeh(**args)
    dataset = cleaner.clean_hf_dataset(dataset, text)
    return dataset 

def write_data_for_train(dataset, text, task = 'cls'):
    data = []
    if task == 'cls':
      for sample in dataset:
          data.append(sample[text])
    elif task == 'ner':
      for sample in dataset:
          data.append(' '.join(sample[text]))
    elif task == 'qa':
      for sample in dataset:
          context, question = text.split(",")
          data.append(sample[context]+" "+sample[question])    
    open(f'data.txt', 'w').write(('\n').join(data))

def create_dataset(config, data_config, vocab_size = 300, 
                   model_name = "birnn", tokenizer_name = "bpe"):

    dataset_name = data_config['name']
    max_tokens = int(config['tokenization']['max_tokens'])
    tok_save_path = config['tokenization']['tok_save_path']

    batch_size = int(config['train']['batch_size'])
    task_name = data_config['task']
    save_dir = config['train']['save_dir']

    # clean and load data
    # load_dataset_kwargs = config['load_dataset_kwargs']
    # dataset = load_dataset(dataset_name,**load_dataset_kwargs)
    try:
        dataset = load_dataset(dataset_name, data_config['subset'])
    except:
        dataset = load_dataset(dataset_name)

    if task_name != 'qa' and task_name != 'mt':
        dataset = clean_dataset(dataset, config, data_config)

    dataset = split_dataset(dataset, data_config)
    examples = copy.deepcopy(dataset)

    if task_name == 'cls':
        # tokenize data
        if 'bert' in model_name:
          tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, model_max_length = 512)
          dataset = dataset.map(lambda examples:tokenizer(examples[data_config['text']], truncation=True, padding='max_length'), batched=True)
          columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        else:
            if tokenizer_name == 'bpe':
                tokenizer = bpe(vocab_size = vocab_size)
            elif tokenizer_name == 'bpe-morph': 
                tokenizer = bpe(vocab_size = vocab_size, morph = True, morph_with_sep=True)

            tok_save_path = f"{save_dir}/{tokenizer.name}/{dataset_name}/"
            if os.path.isfile(f"{tok_save_path}/tok.model"):
                print('loading pretrained tokenizer')
                tokenizer.load(f"{tok_save_path}/")
                dataset = load_from_disk(f'{tok_save_path}/data/')
            else:
                print('training tokenizer from scratch')
                write_data_for_train(dataset['train'], data_config['text'])
                tokenizer.train(file = 'data.txt')
                tokenizer.save(f"{tok_save_path}/")
                dataset = dataset.map(lambda examples:{'input_ids': tokenizer.encode_sentences(examples[data_config['text']], out_length= max_tokens)}, batched=True)
                dataset.save_to_disk(f'{tok_save_path}/data/')                
            columns=['input_ids', 'labels'] 
        
        dataset = dataset.map(lambda examples:{'labels': examples[data_config['label']]}, batched=True)

    elif task_name == 'ner':
        dataset = aggregate_tokens(dataset, config, data_config)
        if 'bert' in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            columns=['input_ids', 'attention_mask', 'labels']
        else:
            if tokenizer_name == 'bpe':
                tokenizer = bpe(vocab_size = vocab_size)
            elif tokenizer_name == 'bpe-morph': 
                tokenizer = bpe(vocab_size = vocab_size, morph = True, morph_with_sep=True)

            tok_save_path = f"{save_dir}/{tokenizer.name}/{dataset_name}/"
            if os.path.isfile(f"{tok_save_path}/tok.model"):
                print('loading pretrained tokenizer')
                tokenizer.load(f"{tok_save_path}/")
                dataset = load_from_disk(f'{tok_save_path}/data/')
            else:
                print('training tokenizer from scratch')
                write_data_for_train(dataset['train'], data_config['text'], task = task_name)
                tokenizer.train(file = 'data.txt')
                tokenizer.save(f"{tok_save_path}/")
                dataset.save_to_disk(f'{tok_save_path}/data/')                

            columns=['input_ids', 'labels'] 
        
        print('aligining the tokens ...')
        for split in dataset:
            dataset[split] = dataset[split].map(lambda x: tokenize_and_align_labels(x, tokenizer, data_config, model_type = model_name)
                                                , batched=True, remove_columns=dataset[split].column_names)
        # dataset = dataset.map(lambda examples:{'labels': examples[data_config['label']]}, batched=True)
    
    elif task_name == 'qa':
        if 'bert' in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        else:
            if tokenizer_name == 'bpe':
                tokenizer = bpe(vocab_size = vocab_size)
            elif tokenizer_name == 'bpe-morph': 
                tokenizer = bpe(vocab_size = vocab_size, morph = True, morph_with_sep=True)

            tok_save_path = f"{save_dir}/{tokenizer.name}/{dataset_name}/"
            if os.path.isfile(f"{tok_save_path}/tok.model"):
                print('loading pretrained tokenizer')
                tokenizer.load(f"{tok_save_path}/")
                dataset = load_from_disk(f'{tok_save_path}/data/')
            else:
                print('training tokenizer from scratch')
                write_data_for_train(dataset['train'], data_config['text'], task = task_name)
                tokenizer.train(file = 'data.txt')
                tokenizer.save(f"{tok_save_path}/")
                dataset.save_to_disk(f'{tok_save_path}/data/')  


        for split in dataset:
          dataset[split] = dataset[split].map(lambda x: prepare_features(x, tokenizer)
                                                , batched=True, remove_columns=dataset[split].column_names)
    elif task_name == 'mt':
        prefix = "translate Romanian to English: "
        source_lang, target_lang = data_config['subset'].split("-")

        if 'bert' in model_name:
             
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            inputs = [prefix + ex[source_lang] for ex in dataset[data_config['text']]]
            targets = [ex[target_lang] for ex in dataset[data_config['text']]]
            dataset = tokenizer(inputs, max_length=128, truncation=True, padding = 'max_length')

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=128, truncation=True, padding = 'max_length')

            dataset["labels"] = labels["input_ids"]
            columns = ['input_ids', 'attention_mask', 'labels']
    #create loaders 
    if task_name != 'qa': 
        for split in dataset:
            dataset[split].set_format(type='torch', columns=columns)
            dataset[split] = torch.utils.data.DataLoader(dataset[split], batch_size=batch_size)
    
    return tokenizer, [dataset['train'], dataset['valid'], dataset['test']], [examples['train'], examples['valid'], examples['test']]
