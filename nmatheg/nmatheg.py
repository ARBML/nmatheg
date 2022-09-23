
import os 
from .dataset import create_dataset
from .models import SimpleClassificationModel, BERTTextClassificationModel\
                    ,BERTTokenClassificationModel,BERTQuestionAnsweringModel\
                    ,SimpleTokenClassificationModel,SimpleQuestionAnsweringModel\
                    ,SimpleMachineTranslationModel,T5MachineTranslationModel
from .configs import create_default_config
import configparser
import json 
from .utils import save_json, get_tokenizer
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer,AutoModelForTokenClassification,AutoModelForQuestionAnswering,AutoModelForSeq2SeqLM
from transformers import pipeline

import torch
from bpe_surgery import bpe 
import numpy as np


class TrainStrategy:
  def __init__(self, datasets, models, tokenizers, vocab_sizes='10000',config_path= None,
               batch_size = 64, epochs = 5, lr = 5e-5, runs = 10, max_tokens = 128, max_train_samples = -1,
               preprocessing = {}):

    if config_path == None:
      self.config = create_default_config(batch_size=batch_size, epochs = epochs, lr = lr, runs = runs,
                                          max_tokens=max_tokens, max_train_samples = max_train_samples, 
                                          preprocessing = preprocessing)
      self.config['dataset'] = {'dataset_name' : datasets}
      self.config['model'] = {'model_name' : models}
      self.config['tokenization']['vocab_size'] = vocab_sizes
      self.config['tokenization']['tokenizer_name'] = tokenizers      
      print(self.config)
    else:
      self.config = configparser.ConfigParser()
      self.config.read(config_path)

    self.datasets_config = configparser.ConfigParser()
    rel_path = os.path.dirname(__file__)
    data_ini_path = os.path.join(rel_path, "datasets.ini")
    self.datasets_config.read(data_ini_path)

  def start(self):
    model_names = [m.strip() for m in self.config['model']['model_name'].split(',')]
    dataset_names = [d.strip() for d in self.config['dataset']['dataset_name'].split(',')]
    tokenizers = [t.strip() for t in self.config['tokenization']['tokenizer_name'].split(',')]
    vocab_sizes = [v.strip() for v in self.config['tokenization']['vocab_size'].split(',')]
    runs = int(self.config['train']['runs'])
    max_tokens = int(self.config['tokenization']['max_tokens'])

    results = {}

    results_path = f"{self.config['train']['save_dir']}/results.json"
    if os.path.isfile(results_path):
      f = open(results_path)
      results = json.load(f)
    print(results)
    for tokenizer_name in tokenizers:
      if not tokenizer_name in results:
        results[tokenizer_name] = {}
      for vocab_size in vocab_sizes:
        if not vocab_size in results[tokenizer_name]:
          results[tokenizer_name][vocab_size] = {}
        for dataset_name in dataset_names:
          if not dataset_name in results[tokenizer_name][vocab_size]:
            results[tokenizer_name][vocab_size][dataset_name] = {} 
          for model_name in model_names:
            if not model_name in results[tokenizer_name][vocab_size][dataset_name]:
              results[tokenizer_name][vocab_size][dataset_name][model_name] = {} 
            for run in range(runs):
              if os.path.isfile(results_path):
                if len(results[tokenizer_name][vocab_size][dataset_name][model_name].keys()) > 0:
                  metric_name = list(results[tokenizer_name][vocab_size][dataset_name][model_name].keys())[0]
                  curr_run = len(results[tokenizer_name][vocab_size][dataset_name][model_name][metric_name])
                  if run < curr_run:
                    print(f"Run {run} already finished ")
                    continue 
              self.data_config = self.datasets_config[dataset_name]
              print(dict(self.data_config))
              task_name = self.data_config['task']
              tokenizer, self.datasets, self.examples = create_dataset(self.config, self.data_config, 
                                                                      vocab_size = int(vocab_size), 
                                                                      model_name = model_name,
                                                                      tokenizer_name = tokenizer_name)
              self.model_config = {'model_name':model_name,
                                  'vocab_size':int(vocab_size),
                                  'num_labels':int(self.data_config['num_labels'])}

              print(self.model_config)
              if task_name == 'cls':                  
                if 'bert' not in model_name:
                  self.model = SimpleClassificationModel(self.model_config)
                else:
                  self.model = BERTTextClassificationModel(self.model_config)
              elif task_name == 'ner':
                if 'bert' not in model_name:
                  self.model = SimpleTokenClassificationModel(self.model_config)
                else:
                  self.model = BERTTokenClassificationModel(self.model_config)

              elif task_name == 'qa':
                if 'bert' not in model_name:
                  self.model = SimpleQuestionAnsweringModel(self.model_config)
                else:
                  self.model = BERTQuestionAnsweringModel(self.model_config)
              elif task_name == 'mt':
                if 'T5' not in model_name:
                  self.model = SimpleMachineTranslationModel(self.model_config, tokenizer = tokenizer)
                else:
                  self.model = T5MachineTranslationModel(self.model_config, tokenizer = tokenizer)
              
              try: new_tokenizer_name = tokenizer.name 
              except: new_tokenizer_name = tokenizer.name_or_path.split('/')[-1]

              self.train_config = {'epochs':int(self.config['train']['epochs']),
                                  'save_dir':f"{self.config['train']['save_dir']}/{new_tokenizer_name}/{dataset_name}/run_{run}",
                                  'batch_size':int(self.config['train']['batch_size']),
                                  'lr':float(self.config['train']['lr']),
                                  'runs':run}
              self.tokenizer_config = {'name': tokenizer_name, 'vocab_size': vocab_size, 'max_tokens': max_tokens,
                                      'save_path': f"{self.config['train']['save_dir']}/{new_tokenizer_name}/{dataset_name}"}
              print(self.tokenizer_config)
              print(self.train_config)
              os.makedirs(self.train_config['save_dir'], exist_ok = True)
              train_dir = f"{self.config['train']['save_dir']}/{new_tokenizer_name}/{dataset_name}/run_{run}"

              if task_name == 'mt':
                metrics = self.model.train(self.datasets, self.examples, **self.train_config) 
              else:
                metrics = self.model.train(self.datasets, self.examples, **self.train_config) 

              save_json(self.train_config, f"{train_dir}/train_config.json")
              save_json(self.data_config, f"{train_dir}/data_config.json")
              save_json(self.model_config, f"{train_dir}/model_config.json")
              save_json(self.tokenizer_config, f"{train_dir}/tokenizer_config.json")
              for metric_name in metrics:
                if metric_name not in results[tokenizer_name][vocab_size][dataset_name][model_name]:
                  results[tokenizer_name][vocab_size][dataset_name][model_name][metric_name] = []
                results[tokenizer_name][vocab_size][dataset_name][model_name][metric_name].append(metrics[metric_name])
              self.model.wipe_memory()
              with open(f"{self.config['train']['save_dir']}/results.json", 'w') as handle:
                json.dump(results, handle)
    return results    

def predict_from_run(save_dir, sentence = "", question = "", context = ""):
  data_config = json.load(open(f"{save_dir}/data_config.json"))
  tokenizer_config = json.load(open(f"{save_dir}/tokenizer_config.json"))
  model_config = json.load(open(f"{save_dir}/model_config.json"))
  model_name = model_config["model_name"]
  task_name = data_config['task']
  tokenizer_name = tokenizer_config["name"]
  tokenizer_save_path = tokenizer_config["save_path"]
  max_tokens = tokenizer_config["max_tokens"]
  vocab_size = tokenizer_config["vocab_size"]

  if model_name == "birnn":
    if task_name == "mt":
      src_tokenizer = get_tokenizer(tokenizer_name, vocab_size = vocab_size)
      trg_tokenizer = get_tokenizer(tokenizer_name, vocab_size = vocab_size)

      src_tokenizer.load(tokenizer_save_path, name = "src_tok")
      trg_tokenizer.load(tokenizer_save_path, name = "trg_tok")

      model = SimpleMachineTranslationModel(model_config, tokenizer = trg_tokenizer)
      model.model.load_state_dict(torch.load(f"{save_dir}/model.pth"))

      encoding = src_tokenizer.encode_sentences([sentence], add_boundry=True, out_length=max_tokens)
      out = model.model(torch.tensor(encoding).to('cuda'), mode = "generate")
      return trg_tokenizer.decode_sentences(out['outputs'])

    elif task_name == "cls":
      tokenizer = get_tokenizer(tokenizer_name, vocab_size = vocab_size)
      tokenizer.load(tokenizer_save_path)

      model = SimpleClassificationModel(model_config)
      model.model.load_state_dict(torch.load(f"{save_dir}/model.pth"))

      encoding = tokenizer.encode_sentences([sentence], add_boundry=True, out_length=max_tokens)
      out = model.model(torch.tensor(encoding).to('cuda'))
      labels = data_config['labels'].split(",")
      return labels[out['logits'].argmax(-1)]
    
    elif task_name == "ner":
      tokenizer = get_tokenizer(tokenizer_name, vocab_size = vocab_size)
      tokenizer.load(tokenizer_save_path)

      model = SimpleTokenClassificationModel(model_config)
      model.model.load_state_dict(torch.load(f"{save_dir}/model.pth"))
      output = []
      labels = data_config['labels'].split(",")
      out_sentence = ""
      sentence_encoding = []
      word_lens = []
      words = sentence.split(' ')
      for word_id , word in enumerate(words):
        enc_words = tokenizer._encode_word(word)
        sentence_encoding += enc_words
        word_lens .append(len(enc_words)) 
      
      while len(sentence_encoding) < max_tokens:
        sentence_encoding.append(0)
      out = model.model(torch.tensor(sentence_encoding).to('cuda'))['logits'].argmax(-1).cpu().numpy()
      i = 0  
      j = 0 
      while i < sum(word_lens):
        preds = out[i:i+word_lens[j]]
        counts = np.bincount(preds)
        mj_label = np.argmax(counts)
        out_sentence += " "+labels[mj_label]
        i += word_lens[j]
        j += 1
      output.append(out_sentence.strip())
      return output
    
    elif task_name == "qa":
      sentence  = question + ' ' + context
      tokenizer = get_tokenizer(tokenizer_name, vocab_size = vocab_size)
      tokenizer.load(tokenizer_save_path)

      model = SimpleQuestionAnsweringModel(model_config)
      model.model.load_state_dict(torch.load(f"{save_dir}/model.pth"))
      encoding = tokenizer.encode_sentences([sentence], out_length=max_tokens)
      out = model.model(torch.tensor(encoding).to('cuda'))
      start_preds = out['start_logits'].argmax(-1).cpu().numpy()
      end_preds = out['end_logits'].argmax(-1).cpu().numpy()      
      return encoding[0][start_preds[0]:end_preds[0]]
  else:
    
    
    if task_name == "cls":
      config = AutoConfig.from_pretrained(model_name)
      model = AutoModelForSequenceClassification.from_pretrained(save_dir, config = config)
      tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, model_max_length = 512)
      encoded_review = tokenizer.encode_plus(
      sentence,
      max_length=512,
      add_special_tokens=True,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      )

      input_ids = encoded_review['input_ids']
      attention_mask = encoded_review['attention_mask']
      output = model(input_ids, attention_mask)
      labels = data_config['labels'].split(",")
      return labels[output['logits'].argmax(-1)]

    elif task_name == "ner":
      labels = data_config['labels'].split(",")
      config = AutoConfig.from_pretrained(model_name, num_labels = 21, id2label = labels)
      tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = 512)
      model = AutoModelForTokenClassification.from_pretrained(save_dir, config = config)
      nlp = pipeline(task_name, model=model, tokenizer=tokenizer)
      return nlp(sentence)

    elif task_name == "qa":
      config = AutoConfig.from_pretrained(model_name)
      tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = 512)
      model = AutoModelForQuestionAnswering.from_pretrained(save_dir, config = config)
      nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)
      return nlp(question=question, context=context)
    
    elif task_name == "mt":
      config = AutoConfig.from_pretrained(model_name)
      tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = 512)
      model = AutoModelForSeq2SeqLM.from_pretrained(save_dir, config = config)
      nlp = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
      return nlp(sentence)
