import tkseem as tk
import bpe_surgery

def get_tokenizer(tok_name, vocab_size = 300):
    tokenizers = {'SentencePieceTokenizer':tk.SentencePieceTokenizer, 'WordTokenizer':tk.WordTokenizer, 'CharacterTokenizer':tk.CharacterTokenizer,
         'MorphologicalTokenizer':tk.MorphologicalTokenizer, 'RandomTokenizer':tk.RandomTokenizer, 'DisjointLetterTokenizer':tk.DisjointLetterTokenizer, 
         'bpe':bpe_surgery.bpe(vocab_size), 'bpe-morph':bpe_surgery.bpe(vocab_size, morph = True, morph_with_sep=True)}
    return tokenizers[tok_name] 

def get_preprocessing_args(config):
    args = {}
    map_bool = {'True':True, 'False':False, '[]': []}
    for key in config['preprocessing']:
        val = config['preprocessing'][key]
        if val in map_bool.keys():
            args[key] = map_bool[val]
        else:
            args[key] = val 
    return args