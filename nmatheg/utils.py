import tkseem as tk
import bpe_surgery

def get_tokenizer(tok_name, vocab_size = 300, lang = 'ar'):
    if tok_name == "WordTokenizer":
      return tk.WordTokenizer(vocab_size=vocab_size)
    elif tok_name == "SentencePieceTokenizer":
      return tk.SentencePieceTokenizer(vocab_size=vocab_size)
    elif tok_name == "CharacterTokenizer":
      return tk.CharacterTokenizer(vocab_size=vocab_size)
    elif tok_name == "RandomTokenizer":
      return tk.RandomTokenizer(vocab_size=vocab_size)
    elif tok_name == "DisjointLetterTokenizer":
      return tk.DisjointLetterTokenizer(vocab_size=vocab_size)
    elif tok_name == "MorphologicalTokenizer":
      return tk.MorphologicalTokenizer(vocab_size=vocab_size)
    else:
      raise('Unrecognized tokenizer name!')

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