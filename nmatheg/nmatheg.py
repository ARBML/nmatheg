
import os 
from .dataset import create_dataset
from .models import SimpleClassificationModel, BERTTextClassificationModel,BERTTokenClassificationModel,BERTQuestionAnsweringModel
from .configs import create_default_config
import pandas as pd
import configparser
import pickle 

class TrainStrategy:
  def __init__(self, config_path = None, datasets = None, models = None, 
               vocab_sizes = None, tokenizers = None, **kwargs):
    if config_path == None:
      self.config = create_default_config(**kwargs)
      if datasets:
        self.config['dataset'] = {'dataset_name' : datasets}
      if models:
        self.config['model'] = {'model_name' : models}
      if vocab_sizes:
        self.config['tokenization']['vocab_size'] = vocab_sizes
      if tokenizers:
        self.config['Tokenization']['tokenizer_name'] = tokenizers
    else:
      self.config = configparser.ConfigParser()
      self.config.read(config_path)

    self.data_config = configparser.ConfigParser()
    rel_path = os.path.dirname(__file__)
    data_ini_path = os.path.join(rel_path, "datasets.ini")
    self.data_config.read(data_ini_path)

  def start(self):
    model_names = [m.strip() for m in self.config['model']['model_name'].split(',')]
    dataset_names = [d.strip() for d in self.config['dataset']['dataset_name'].split(',')]
    tokenizers = [t.strip() for t in self.config['Tokenization']['tokenizer_name'].split(',')]
    vocab_sizes = [int(v.strip()) for v in self.config['tokenization']['vocab_size'].split(',')]
    runs = int(self.config['train']['runs'])
    output = []
    dataset_metrics = []

    for tokenizer_name in tokenizers:
      for vocab_size in vocab_sizes:
        for dataset_name in dataset_names: 
          for model_name in model_names:
            for run in range(runs):
              task_name = self.data_config[dataset_name]['task']
              tokenizer, self.datasets, self.examples = create_dataset(self.config, self.data_config, 
                                                                      vocab_size = vocab_size, model_name = model_name,
                                                                      tokenizer_name = tokenizer_name)
              self.model_config = {'model_name':model_name,
                                  'vocab_size':vocab_size,
                                  'num_labels':int(self.data_config[dataset_name]['num_labels'])}

              print(self.model_config)
              if task_name == 'cls':
                if 'bert' in model_name:
                  self.model = BERTTextClassificationModel(self.model_config)
                elif 'birnn' in model_name:
                  self.model = SimpleClassificationModel(self.model_config)
                else:
                  raise('error not recognized model name')
              elif task_name == 'ner':
                self.model = BERTTokenClassificationModel(self.model_config)

              elif task_name == 'qa':
                self.model = BERTQuestionAnsweringModel(self.model_config)
              self.train_config = {'epochs':int(self.config['train']['epochs']),
                                  'save_dir':f"{self.config['train']['save_dir']}/{tokenizer.name}/{dataset_name}/run_{run}",
                                  'batch_size':int(self.config['train']['batch_size']),
                                  'lr':float(self.config['train']['lr']),
                                  'runs':run}
              print(self.train_config)
              os.makedirs(self.train_config['save_dir'], exist_ok = True)
              metrics = self.model.train(self.datasets, self.examples, **self.train_config) 

              for metric_name in metrics:
                if model_name == model_names[0]:
                  dataset_metrics.append(dataset_name+metric_name)
              output.append([model_name, dataset_name, tokenizer_name, run,  metrics])
              self.model.wipe_memory()
    
    
    model_results = {model_name:[0]*len(dataset_metrics) for model_name in model_names}
    metric_names = ['Model']
    dataset_names = ['']
    with open(f"{self.config['train']['save_dir']}/{tokenizer.name}/results.pl", 'wb') as handle:
      pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for row in output:
      model_name = row[0]
      dataset_name = row[1]
      metrics = row[2]
      task_name = self.data_config[dataset_name]['task']
      for metric_name in row[2]: 
        model_results[model_name][dataset_metrics.index(dataset_name+metric_name)] = round(metrics[metric_name]*100, 2)
        
        # this is only used once 
        if model_name == model_names[0]:
          metric_names.append(metric_name) 
          dataset_names.append(dataset_name) 
    
    rows = []
    for model_name in model_results:
      rows.append([model_name]+model_results[model_name])

    return pd.DataFrame(rows, columns = [dataset_names, metric_names])     
