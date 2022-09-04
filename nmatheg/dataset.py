
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

def split_dataset(dataset, data_config, seed = 42, max_train_samples = -1):
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

    if max_train_samples < len(dataset['train']):
        print(f"truncating train samples from {len(dataset['train'])} to {max_train_samples}")
        dataset['train'] = dataset['train'].select(range(max_train_samples))
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
    elif task == 'mt':
      for sample in dataset:
          context, question = text.split(",")
          data.append(sample[context]+" "+sample[question])

    open(f'data.txt', 'w').write(('\n').join(data))

def create_dataset(config, data_config, vocab_size = 300, 
                   model_name = "birnn", tokenizer_name = "bpe"):

    dataset_name = data_config['name']
    max_tokens = int(config['tokenization']['max_tokens'])
    tok_save_path = config['tokenization']['tok_save_path']
    max_train_samples = int(config['tokenization']['max_train_samples'])

    batch_size = int(config['train']['batch_size'])
    task_name = data_config['task']
    save_dir = config['train']['save_dir']

    # clean and load data
    # load_dataset_kwargs = config['load_dataset_kwargs']
    # dataset = load_dataset(dataset_name,**load_dataset_kwargs)
    try:
        if dataset_name == "tatoeba_mt":
            print(f"downloading {dataset_name} manually ... ")
            clone = "git lfs clone https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt"
            wget = "wget https://huggingface.co/datasets/Zaid/tatoeba_mt/raw/main/tatoeba_mt.py -O tatoeba_mt/tatoeba_mt.py"

            os.system(f"{clone} && {wget}")

        dataset = load_dataset(dataset_name, data_config['subset'])
    except:
        dataset = load_dataset(dataset_name)
    
    print(dataset)
    if task_name != 'qa' and task_name != 'mt':
        dataset = clean_dataset(dataset, config, data_config)

    dataset = split_dataset(dataset, data_config, max_train_samples=max_train_samples)
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
        prefix = "translate English to Arabic: "
        trg_lang, src_lang = data_config['text'].split(",")

        if 't5' in model_name:
             
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            def preprocess(dataset):
                inputs = [prefix + ex for ex in dataset[src_lang]]
                targets = [ex for ex in dataset[trg_lang]]
                dataset = tokenizer(inputs, max_length=128, truncation=True, padding = 'max_length')

                # Setup the tokenizer for targets
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(targets, max_length=128, truncation=True, padding = 'max_length')

                dataset["labels"] = labels["input_ids"]
                return dataset
            dataset = dataset.map(preprocess, batched=True)
            columns = ['input_ids', 'attention_mask', 'labels']
        else:
            if tokenizer_name == 'bpe':
                src_tokenizer = bpe(vocab_size = vocab_size, lang = 'en')
                trg_tokenizer = bpe(vocab_size = vocab_size, lang = 'ar')
            elif tokenizer_name == 'bpe-morph':
                src_tokenizer = bpe(vocab_size = vocab_size, lang = 'en') 
                trg_tokenizer = bpe(vocab_size = vocab_size, morph = True, morph_with_sep=True, lang = 'ar')

            tok_save_path = f"{save_dir}/{trg_tokenizer.name}/{dataset_name}/"

            if os.path.isfile(f"{tok_save_path}/trg_tok.model"):
                print('loading pretrained tokenizers')
                src_tokenizer.load(f"{tok_save_path}/", name = "src_tok")
                trg_tokenizer.load(f"{tok_save_path}/", name = "trg_tok")
                dataset = load_from_disk(f'{tok_save_path}/data/')
            else:
                print('training tokenizer from scratch')
                open('src_data.txt', 'w').write('\n'.join(dataset['train'][src_lang]))
                open('trg_data.txt', 'w').write('\n'.join(dataset['train'][trg_lang]))

                src_tokenizer.train(file = 'src_data.txt')
                trg_tokenizer.train(file = 'trg_data.txt')
                src_tokenizer.save(f"{tok_save_path}/", name = 'src_tok')
                trg_tokenizer.save(f"{tok_save_path}/", name = 'trg_tok')


                def preprocess(dataset):
                    inputs = [ex for ex in dataset[src_lang]]
                    targets = [ex for ex in dataset[trg_lang]]
                    
                    input_ids = src_tokenizer.encode_sentences(inputs, out_length = max_tokens, add_boundry = True)
                    labels = trg_tokenizer.encode_sentences(targets, out_length = max_tokens, add_boundry = True)
                    dataset = dataset.add_column("input_ids", input_ids)
                    dataset = dataset.add_column("labels", labels)
                    return dataset
                
                for split in dataset: 
                    dataset[split] = preprocess(dataset[split]) 
                
                
                dataset.save_to_disk(f'{tok_save_path}/data/')  

            columns = ['input_ids', 'labels']
            tokenizer = trg_tokenizer
            
    #create loaders 
    if task_name != 'qa': 
        for split in dataset:
            dataset[split].set_format(type='torch', columns=columns)
            dataset[split] = torch.utils.data.DataLoader(dataset[split], batch_size=batch_size)
    
    return tokenizer, [dataset['train'], dataset['valid'], dataset['test']], [examples['train'], examples['valid'], examples['test']]
