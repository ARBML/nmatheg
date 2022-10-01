import tkseem as tk
try:
  import bpe_surgery
except:
  pass
import json 

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
    elif tok_name == "bpe":
      return bpe_surgery.bpe(vocab_size=vocab_size)
    elif tok_name == "bpe-morph":
      return bpe_surgery.bpe(vocab_size=vocab_size, morph=True, morph_with_sep=True)
    elif tok_name == "bpe-seg":
      return bpe_surgery.bpe(vocab_size=vocab_size, seg = True)
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

def save_json(ob, save_path):
  with open(save_path, 'w') as handle:
    json.dump(dict(ob), handle)