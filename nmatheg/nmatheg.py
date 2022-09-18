
import os 
from .dataset import create_dataset
from .models import SimpleClassificationModel, BERTTextClassificationModel\
                    ,BERTTokenClassificationModel,BERTQuestionAnsweringModel\
                    ,SimpleTokenClassificationModel,SimpleQuestionAnsweringModel\
                    ,SimpleMachineTranslationModel,T5MachineTranslationModel
from .configs import create_default_config
import configparser
import json 
from .utils import save_json

class TrainStrategy:
  def __init__(self, datasets, models, tokenizers, vocab_sizes='10000',config_path= None,
               batch_size = 64, epochs = 5, lr = 5e-5, runs = 10, max_tokens = 128, max_train_samples = -1):

    if config_path == None:
      self.config = create_default_config(batch_size=batch_size, epochs = epochs, lr = lr, runs = runs,
                                          max_tokens=max_tokens, max_train_samples = max_train_samples)
      self.config['dataset'] = {'dataset_name' : datasets}
      self.config['model'] = {'model_name' : models}
      self.config['tokenization']['vocab_size'] = vocab_sizes
      self.config['tokenization']['tokenizer_name'] = tokenizers
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
    output = []
    results = {}
    dataset_metrics = []

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
              except: new_tokenizer_name = tokenizer.name_or_path.split('/')[0]

              self.train_config = {'epochs':int(self.config['train']['epochs']),
                                  'save_dir':f"{self.config['train']['save_dir']}/{new_tokenizer_name}/{dataset_name}/run_{run}",
                                  'batch_size':int(self.config['train']['batch_size']),
                                  'lr':float(self.config['train']['lr']),
                                  'runs':run}
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

              for metric_name in metrics:
                if metric_name not in results[tokenizer_name][vocab_size][dataset_name][model_name]:
                  results[tokenizer_name][vocab_size][dataset_name][model_name][metric_name] = []
                results[tokenizer_name][vocab_size][dataset_name][model_name][metric_name].append(metrics[metric_name])
              self.model.wipe_memory()
              with open(f"{self.config['train']['save_dir']}/results.json", 'w') as handle:
                json.dump(results, handle)
    return results    
