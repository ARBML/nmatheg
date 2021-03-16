import tkseem as tk

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