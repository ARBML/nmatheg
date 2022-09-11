import tkseem as tk
import bpe_surgery

def get_tokenizer(tok_name, vocab_size = 300, lang = 'ar'):
    tokenizers = {'bpe':bpe_surgery.bpe(vocab_size, lang = lang), 
                  'bpe-morph':bpe_surgery.bpe(vocab_size, morph = True, morph_with_sep=True, lang = lang),
                  'bpe-seg':bpe_surgery.bpe(vocab_size, seg = True, lang = lang)}
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