
import os 
from .dataset import create_dataset
from .models import SimpleClassificationModel, BERTTextClassificationModel\
                    ,BERTTokenClassificationModel,BERTQuestionAnsweringModel\
                    ,SimpleTokenClassificationModel,SimpleQuestionAnsweringModel\
                    ,SimpleMachineTranslationModel,T5MachineTranslationModel
from .configs import create_default_config
import configparser
import pickle 

class TrainStrategy:
  def __init__(self, datasets, models, tokenizers, vocab_sizes='10000',
               config_path= None,  batch_size = 64, epochs = 5, lr = 5e-5, runs = 10):

    if config_path == None:
      self.config = create_default_config(batch_size=batch_size, epochs = epochs, lr = lr, runs = runs)
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
    vocab_sizes = [int(v.strip()) for v in self.config['tokenization']['vocab_size'].split(',')]
    runs = int(self.config['train']['runs'])
    output = []
    dataset_metrics = []

    for tokenizer_name in tokenizers:
      for vocab_size in vocab_sizes:
        for dataset_name in dataset_names: 
          for model_name in model_names:
            for run in range(runs):
              self.data_config = self.datasets_config[dataset_name]
              print(dict(self.data_config))
              task_name = self.data_config['task']
              tokenizer, self.datasets, self.examples = create_dataset(self.config, self.data_config, 
                                                                      vocab_size = vocab_size, 
                                                                      model_name = model_name,
                                                                      tokenizer_name = tokenizer_name)
              self.model_config = {'model_name':model_name,
                                  'vocab_size':vocab_size,
                                  'num_labels':int(self.data_config['num_labels'])}

              print(self.model_config)
              if task_name == 'cls':                  
                if 'birnn' in model_name:
                  self.model = SimpleClassificationModel(self.model_config)
                else:
                  self.model = BERTTextClassificationModel(self.model_config)
              elif task_name == 'ner':
                if 'birnn' in model_name:
                  self.model = SimpleTokenClassificationModel(self.model_config)
                else:
                  self.model = BERTTokenClassificationModel(self.model_config)

              elif task_name == 'qa':
                if 'birnn' in model_name:
                  self.model = SimpleQuestionAnsweringModel(self.model_config)
                else:
                  self.model = BERTQuestionAnsweringModel(self.model_config)
              elif task_name == 'mt':
                if 'seq2seq' in model_name:
                  self.model = SimpleMachineTranslationModel(self.model_config)
                else:
                  self.model = T5MachineTranslationModel(self.model_config)
              
              try: tokenizer_name = tokenizer.name 
              except: tokenizer_name = tokenizer.name_or_path.split('/')[0]

              self.train_config = {'epochs':int(self.config['train']['epochs']),
                                  'save_dir':f"{self.config['train']['save_dir']}/{tokenizer_name}/{dataset_name}/run_{run}",
                                  'batch_size':int(self.config['train']['batch_size']),
                                  'lr':float(self.config['train']['lr']),
                                  'runs':run}
              print(self.train_config)
              os.makedirs(self.train_config['save_dir'], exist_ok = True)
              if task_name == 'mt':
                metrics = self.model.train(self.datasets, self.examples, tokenizer, **self.train_config) 
              else:
                metrics = self.model.train(self.datasets, self.examples, **self.train_config) 

              for metric_name in metrics:
                if model_name == model_names[0]:
                  dataset_metrics.append(dataset_name+metric_name)
              output.append([model_name, dataset_name, tokenizer_name, run,  metrics])
              self.model.wipe_memory()
    with open(f"{self.config['train']['save_dir']}/results.pl", 'wb') as handle:
      pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output    
