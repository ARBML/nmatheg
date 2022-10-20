
from genericpath import isdir
from regex import E
import tnkeeh as tn 
from datasets import load_dataset, load_from_disk
try:
  import bpe_surgery
except:
  pass

import os 
from .utils import get_preprocessing_args, get_tokenizer
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

    if max_train_samples < len(dataset['train']) and max_train_samples != -1:
        print(f"truncating train samples from {len(dataset['train'])} to {max_train_samples}")
        dataset['train'] = dataset['train'].select(range(max_train_samples))

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
    
    columns = list(dataset.keys())
    for key in columns: 
        if key not in split_names:
            del dataset[key]
    return dataset 


def clean_dataset(dataset, config, data_config, task = 'cls'):
    if task == 'mt':
        sourceString, targetString = data_config['text'].split(',')
        args = get_preprocessing_args(config)
        cleaner = tn.Tnkeeh(**args)
        dataset = cleaner.clean_hf_dataset(dataset, targetString)
        return dataset
    elif task == 'qa':
        question, context = data_config['text'].split(',')
        args = get_preprocessing_args(config)
        cleaner = tn.Tnkeeh(**args)
        dataset = cleaner.clean_hf_dataset(dataset, question)
        return dataset
    elif task == 'nli':
        premise, hypothesis = data_config['text'].split(',')
        args = get_preprocessing_args(config)
        cleaner = tn.Tnkeeh(**args)
        dataset = cleaner.clean_hf_dataset(dataset, premise)
        dataset = cleaner.clean_hf_dataset(dataset, hypothesis)
        return dataset
    else:
        args = get_preprocessing_args(config)
        cleaner = tn.Tnkeeh(**args)
        dataset = cleaner.clean_hf_dataset(dataset, data_config['text'])
        return dataset 

def write_data_for_train(dataset, text, path, task = 'cls'):
    data = []
    if task == 'cls':
      for sample in dataset:
          data.append(sample[text])
    elif task == 'nli':
      for sample in dataset:
          hypothesis, premise = text.split(",")
          data.append(sample[hypothesis]+" "+sample[premise])
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

    open(f'{path}/data.txt', 'w').write(('\n').join(data))

def get_prev_tokenizer(save_dir, tokenizer_name, vocab_size, dataset_name, model_name):
    prev_vocab_sizes = [int(v) for v in os.listdir(f"{save_dir}/{tokenizer_name}") if int(v) != vocab_size and dataset_name in os.listdir(f"{save_dir}/{tokenizer_name}/{v}")]

    if len(prev_vocab_sizes) == 0:
        return ""
    else:
        return f"{save_dir}/{tokenizer_name}/{max(prev_vocab_sizes)}/{dataset_name}/{model_name}/tokenizer"

def create_dataset(config, data_config, vocab_size = 300, 
                   model_name = "birnn", tokenizer_name = "bpe", clean = True):

    hf_dataset_name = data_config['name']
    dataset_name = hf_dataset_name.split("/")[-1] #in case we have / in the name
    max_tokens = int(config['tokenization']['max_tokens'])
    tok_save_path = config['tokenization']['tok_save_path']
    max_train_samples = int(config['tokenization']['max_train_samples'])

    batch_size = int(config['train']['batch_size'])
    task_name = data_config['task']
    save_dir = config['train']['save_dir']
    tok_save_path = f"{save_dir}/{tokenizer_name}/{vocab_size}/{dataset_name}/{model_name}/tokenizer"
    data_save_path = f"{save_dir}/{tokenizer_name}/{vocab_size}/{dataset_name}/{model_name}/data"

    prev_tok_save_path = get_prev_tokenizer(save_dir, tokenizer_name, vocab_size, dataset_name, model_name)
    
    # clean and load data
    # load_dataset_kwargs = config['load_dataset_kwargs']
    # dataset = load_dataset(dataset_name,**load_dataset_kwargs)
    if 'subset' in data_config:
        dataset = load_dataset(hf_dataset_name, data_config['subset'])
    else:
        dataset = load_dataset(hf_dataset_name)
    
    if task_name != "qa" and clean:
      dataset = clean_dataset(dataset, config, data_config, task = task_name)

    dataset = split_dataset(dataset, data_config, max_train_samples=max_train_samples)
    examples = copy.deepcopy(dataset)
    print(dataset)
    if 'birnn' in model_name:
      model_type = 'rnn'
    else:
      model_type = 'transformer'
    if task_name == 'cls':
        # tokenize data
        if 'birnn' not in model_name:
          tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, model_max_length = 512)
          if not os.path.isfile(f"{data_save_path}/dataset_dict.json"):
            dataset = dataset.map(lambda examples:tokenizer(examples[data_config['text']], truncation=True, padding='max_length'), batched=True)
            dataset = dataset.map(lambda examples:{'labels': examples[data_config['label']]}, batched=True)
            dataset.save_to_disk(data_save_path)
          else:
            dataset = load_from_disk(data_save_path)
          columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        else:
            tokenizer = get_tokenizer(tokenizer_name, vocab_size= vocab_size)

            if os.path.isfile(f"{tok_save_path}/tok.model"):
                print('loading pretrained tokenizer')
                tokenizer.load(tok_save_path)
                dataset = load_from_disk(data_save_path)
            else:
                write_data_for_train(dataset['train'], data_config['text'], data_save_path)
                if prev_tok_save_path != "":
                    tokenizer.load(prev_tok_save_path)
                else:
                    print('training tokenizer from scratch')
                tokenizer.train(file_path = f'{data_save_path}/data.txt')
                tokenizer.save(tok_save_path)
                dataset = dataset.map(lambda examples:{'input_ids': tokenizer.encode_sentences(examples[data_config['text']], out_length= max_tokens)}, batched=True)
                dataset = dataset.map(lambda examples:{'labels': examples[data_config['label']]}, batched=True)
                dataset.save_to_disk(data_save_path)                
            columns=['input_ids', 'labels']
             
    elif task_name == 'nli':
        # tokenize data
        premise, hypothesis = data_config['text'].split(",")
        if 'birnn' not in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, model_max_length = 512)
            def concat(examples):
                texts = (examples[premise], examples[hypothesis])
                result = tokenizer(*texts, truncation=True, padding='max_length')
                return result
            
            if not os.path.isfile(f"{data_save_path}/dataset_dict.json"):
                dataset = dataset.map(concat, batched=True)
                dataset.save_to_disk(data_save_path)
            else:
                load_dataset(data_save_path)
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        else:
            tokenizer = get_tokenizer(tokenizer_name, vocab_size= vocab_size)
            if os.path.isfile(f"{tok_save_path}/tok.model"):
                print('loading pretrained tokenizer')
                tokenizer.load(tok_save_path)
                dataset = load_from_disk(data_save_path)
            else:
                
                write_data_for_train(dataset['train'], data_config['text'], data_save_path, task = 'nli')
                if prev_tok_save_path != "":
                    tokenizer.load(prev_tok_save_path)
                else:
                    print('training tokenizer from scratch')
                tokenizer.train(file_path = f"{data_save_path}/data.txt")
                tokenizer.save(tok_save_path)

                def concat(example):
                  example["text"] = example[premise] + ' ' + example[hypothesis]
                  return example

                dataset = dataset.map(lambda examples:{'input_ids': tokenizer.encode_sentences(sentences1 = examples[premise], sentences2 = examples[hypothesis], out_length= max_tokens)}, batched=True)
                dataset = dataset.map(lambda examples:{'labels': examples[data_config['label']]}, batched=True)
                dataset.save_to_disk(data_save_path)                
            columns=['input_ids', 'labels']

    elif task_name == 'ner':
        dataset = aggregate_tokens(dataset, config, data_config)
        if 'birnn' not in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if not os.path.isfile(f"{data_save_path}/dataset_dict.json"):
                print('aligining the tokens ...')
                for split in dataset:
                    dataset[split] = dataset[split].map(lambda x: tokenize_and_align_labels(x, tokenizer, data_config, model_type = model_type)
                                                        , batched=True, remove_columns=dataset[split].column_names)
                dataset.save_to_disk(data_save_path)
            else:
                dataset = load_from_disk(data_save_path)
            columns=['input_ids', 'attention_mask', 'labels']
        else:
            tokenizer = get_tokenizer(tokenizer_name, vocab_size= vocab_size)

            if  os.path.isfile(f"{tok_save_path}/tok.model"):
                print('loading pretrained tokenizer')
                tokenizer.load(tok_save_path)
                dataset = load_from_disk(data_save_path)
            else:
                write_data_for_train(dataset['train'], data_config['text'], data_save_path, task = task_name)
                if prev_tok_save_path != "":
                    tokenizer.load(prev_tok_save_path)
                else:
                    print('training tokenizer from scratch')
                tokenizer.train(file_path = f'{data_save_path}/data.txt')
                tokenizer.save(tok_save_path)
                print('aligining the tokens ...')
                for split in dataset:
                    dataset[split] = dataset[split].map(lambda x: tokenize_and_align_labels(x, tokenizer, data_config, model_type = model_type)
                                                        , batched=True, remove_columns=dataset[split].column_names)
                dataset.save_to_disk(data_save_path)                

            columns=['input_ids', 'labels'] 
        
    elif task_name == 'qa':
        if 'birnn' not in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if not os.path.isfile(f"{data_save_path}/dataset_dict.json"):
                for split in dataset:
                        dataset[split] = dataset[split].map(lambda x: prepare_features(x, tokenizer, data_config, model_type = model_type, max_len = max_tokens)
                                                    , batched=True, remove_columns=dataset[split].column_names)
                dataset.save_to_disk(data_save_path)
            else:
                dataset = load_from_disk(data_save_path)
            columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions']
        else:
            tokenizer = get_tokenizer(tokenizer_name, vocab_size= vocab_size)

            if os.path.isfile(f"{tok_save_path}/tok.model"):
                print('loading pretrained tokenizer')
                tokenizer.load(tok_save_path)
                dataset = load_from_disk(data_save_path)
            else:
                write_data_for_train(dataset['train'], data_config['text'], data_save_path, task = task_name)
                if prev_tok_save_path != "":
                    tokenizer.load(prev_tok_save_path)
                else:
                    print('training tokenizer from scratch')
                tokenizer.train(file_path = f'{data_save_path}/data.txt')
                tokenizer.save(tok_save_path)
                for split in dataset:
                    dataset[split] = dataset[split].map(lambda x: prepare_features(x, tokenizer, data_config, model_type = model_type, max_len = max_tokens)
                                                , batched=True, remove_columns=dataset[split].column_names)
                dataset.save_to_disk(data_save_path)
            columns=['input_ids', 'start_positions', 'end_positions']  

    elif task_name == 'mt':
        prefix = "translate English to Arabic: "
        src_lang, trg_lang = data_config['text'].split(",")

        if 'birnn' not in model_name:
             
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
            if not os.path.isfile(f"{data_save_path}/dataset_dict.json"):
                dataset = dataset.map(preprocess, batched=True)
                dataset.save_to_disk(data_save_path)
            else:
                dataset = load_from_disk(data_save_path)
            columns = ['input_ids', 'attention_mask', 'labels']
        else:
            src_tokenizer = get_tokenizer('bpe', vocab_size= 1000)
            trg_tokenizer = get_tokenizer(tokenizer_name, vocab_size= vocab_size)
            src_tok_save_path = f"{save_dir}/{tokenizer_name}/1000/{dataset_name}/{model_name}/tokenizer"

            if os.path.isfile(f"{tok_save_path}/trg_tok.model"):
                print('loading pretrained tokenizers')
                src_tokenizer.load(f"{src_tok_save_path}/", name = "src_tok")
                trg_tokenizer.load(f"{tok_save_path}/", name = "trg_tok")
                dataset = load_from_disk(f'{tok_save_path}/data/')
            else:
                open(f'{data_save_path}/src_data.txt', 'w').write('\n'.join(dataset['train'][src_lang]))
                open(f'{data_save_path}/trg_data.txt', 'w').write('\n'.join(dataset['train'][trg_lang]))

                if not os.path.isfile(f"{src_tok_save_path}/src_tok.model"):
                    src_tokenizer.train(file_path = f'{data_save_path}/src_data.txt')
                    src_tokenizer.save(f"{tok_save_path}/", name = 'src_tok')

                if prev_tok_save_path != "":
                    tokenizer.load(prev_tok_save_path)
                else:
                    print('training tokenizer from scratch')

                trg_tokenizer.train(file_path = f'{data_save_path}/trg_data.txt')
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
            dataset[split] = torch.utils.data.DataLoader(dataset[split], batch_size=batch_size, shuffle = True)
    
    return tokenizer, [dataset['train'], dataset['valid'], dataset['test']], [examples['train'], examples['valid'], examples['test']]
