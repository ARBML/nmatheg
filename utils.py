import tkseem as tk
import configparser
import tnkeeh as tn 
from datasets import load_dataset
import os 

def get_tokenizer(tok_name):
    tokenizers = {'SentencePieceTokenizer':tk.SentencePieceTokenizer, 'WordTokenizer':tk.WordTokenizer, 'CharacterTokenizer':tk.CharacterTokenizer,
         'MorphologicalTokenizer':tk.MorphologicalTokenizer, 'RandomTokenizer':tk.RandomTokenizer, 'DisjointLetterTokenizer':tk.DisjointLetterTokenizer}
    return tokenizers[tok_name] 

def get_preprocessing_args(config):
    args = {}
    map_bool = {'True':True, 'False':False, '[]': []}
    for key in config['preprocessing']:
        val = config['preprocessing'][key]
        args[key] = map_bool[val]
    return args

def write_split(dataset_name, config, split = ('train', 'train[:100%]')):
    data = []
    lbls = []

    data_config = configparser.ConfigParser()
    data_config.read('datasets.ini')
    
    text = data_config[dataset_name]['text']
    label = data_config[dataset_name]['label']

    args = get_preprocessing_args(config)
    cleaner = tn.Tnqeeh(**args)
    dataset = load_dataset(dataset_name, split=split[1])
    dataset = cleaner.clean_hf_dataset(dataset, text)

    for sample in dataset:
        data.append(sample[text])
        lbls.append(str(sample[label]))
    
    open(f'data/{split[0]}_data.txt', 'w').write(('\n').join(data))
    open(f'data/{split[0]}_lbls.txt', 'w').write(('\n').join(lbls))

def write_dataset(dataset_name, config, has_no_split = True):
    os.makedirs('data', exist_ok=True)
    if has_no_split:
        for split_name , ratio in [('train', 'train[0%:80%]'), ('valid', 'train[80%:90%]'), ('test', 'train[-10%:]')]:
            write_split(dataset_name, config, split = (split_name, ratio))