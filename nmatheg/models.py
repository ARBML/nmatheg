from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup)
import os
import time
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_metric
import copy 
from .ner_utils import get_labels

class BiRNN(nn.Module):
    def __init__(self, vocab_size, num_labels):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, 128)
        self.bigru = nn.GRU(128, 256, bidirectional=True)
        self.fc = nn.Linear(512, num_labels)
        self.num_labels = num_labels
        
    def forward(self, 
                input_ids,
                labels):

        embedded = self.embedding(input_ids)        
        out = self.bigru(embedded)
        logits = self.fc(out[0][:,0,:])
        loss = self.compute_loss(logits, labels)
        return {'loss':loss,
                'logits':logits} 
    
    def compute_loss(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels)
        return loss

class BaseTextClassficationModel:
    def __init__(self, config):
        self.model = nn.Module()
        self.num_labels = config['num_labels']
        self.model_name = config['model_name']
        self.vocab_size = config['vocab_size']

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def train(self, datasets, examples, **kwargs):
        save_dir = kwargs['save_dir']
        epochs = kwargs['epochs']
        lr = kwargs['lr']

        train_dataset, valid_dataset, test_dataset = datasets 

        self.optimizer = AdamW(self.model.parameters(), lr = lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                        num_warmup_steps = 0,
                                                        num_training_steps=len(train_dataset)//epochs)
        filepath = os.path.join(save_dir, 'model.pth')
        best_accuracy = 0 
        for epoch in range(epochs):
            accuracy = 0 
            loss = 0 
            self.model.train().to(self.device)
            for _, batch in enumerate(train_dataset):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs['loss']
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()  
                self.optimizer.zero_grad()

                labels = batch['labels'].cpu() 
                preds = outputs['logits'].argmax(-1).cpu() 
                accuracy += accuracy_score(labels, preds) /len(train_dataset)
                loss += loss / len(train_dataset)
                batch = None 
            print(f"Epoch {epoch} Train Loss {loss:.4f} Train Accuracy {accuracy:.4f}")
            
            self.model.eval().to(self.device)
            results = self.evaluate_dataset(valid_dataset)
            print(f"Epoch {epoch} Valid Loss {results['loss']:.4f} Valid Accuracy {results['accuracy']:.4f}")

            val_accuracy = results['accuracy']
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), filepath)

            #Later to restore:
        
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        test_metrics = self.evaluate_dataset(test_dataset)
        print(f"Test Loss {test_metrics['loss']:.4f} Test Accuracy {test_metrics['accuracy']:.4f}")
        return {'accuracy':test_metrics['accuracy']} 
    
    def evaluate_dataset(self, dataset):
        accuracy = 0
        loss = 0 
        for _, batch in enumerate(dataset):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs['loss']
            labels = batch['labels'].cpu() 
            preds = outputs['logits'].argmax(-1).cpu() 
            accuracy += accuracy_score(labels, preds) /len(dataset)
            loss += loss / len(dataset)
            batch = None 
        return {'loss':float(loss.cpu().detach().numpy()), 'accuracy':accuracy}

class SimpleClassificationModel(BaseTextClassficationModel):
    def __init__(self, config):
        BaseTextClassficationModel.__init__(self, config)
        self.model = BiRNN(self.vocab_size, self.num_labels)
        self.model.to(self.device)    
        self.optimizer = AdamW(self.model.parameters(), lr = 5e-5)

    def wipe_memory(self):
        self.model = None  
        self.optimizer = None 
        torch.cuda.empty_cache()

class BERTTextClassificationModel(BaseTextClassficationModel):
    def __init__(self, config):
        BaseTextClassficationModel.__init__(self, config)
        config = AutoConfig.from_pretrained(self.model_name,num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config = config)
    
    def wipe_memory(self):
        self.model = None  
        self.optimizer = None 
        torch.cuda.empty_cache()

class BaseTokenClassficationModel:
    def __init__(self, config):
        self.model = nn.Module()
        self.num_labels = config['num_labels']
        self.model_name = config['model_name']
        self.vocab_size = config['vocab_size']
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.metric  = load_metric("seqeval")
        self.accelerator = Accelerator()

    def train(self, datasets, examples, **kwargs):
        save_dir = kwargs['save_dir']
        epochs = kwargs['epochs']
        lr = kwargs['lr']
        self.optimizer = AdamW(self.model.parameters(), lr = lr)

        train_dataset, valid_dataset, test_dataset = datasets 
        filepath = os.path.join(save_dir, 'model.pth')
        best_accuracy = 0 
        for epoch in range(epochs):
            accuracy = 0 
            loss = 0 
            self.model.train().to(self.device)
            predictions , true_labels = [], []
            for _, batch in enumerate(train_dataset):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs['loss']
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss += loss / len(train_dataset)
                batch = None

            train_metrics = self.evaluate_dataset(train_dataset) 
            print(f"Epoch {epoch} Valid Loss {train_metrics['loss']:.4f} Valid F1 {train_metrics['f1']:.4f}")
            
            self.model.eval().to(self.device)
            valid_metrics = self.evaluate_dataset(valid_dataset)
            print(f"Epoch {epoch} Valid Loss {valid_metrics['loss']:.4f} Valid F1 {valid_metrics['f1']:.4f}")

            val_accuracy = valid_metrics['f1']
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), filepath)
        
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        test_metrics = self.evaluate_dataset(test_dataset)
        print(f"Test Loss {test_metrics['loss']:.4f} Test F1 {test_metrics['f1']:.4f}")
        return {
                    "precision": test_metrics["precision"],
                    "recall": test_metrics["recall"],
                    "f1": test_metrics["f1"],
                    "accuracy": test_metrics["accuracy"],
                }
        
    def evaluate_dataset(self, dataset):
        accuracy = 0
        loss = 0 
        for _, batch in enumerate(dataset):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs['loss']
            labels = batch['labels']
            predictions = outputs['logits'].argmax(dim=-1)
            
            predictions_gathered = self.accelerator.gather(predictions)
            labels_gathered = self.accelerator.gather(labels)
            preds, refs = get_labels(predictions_gathered, labels_gathered)
            self.metric.add_batch(
                predictions=preds,
                references=refs,
            )

            loss += loss / len(dataset)
            batch = None
        results = self.metric.compute()

        return {
                    "loss":float(loss.cpu().detach().numpy()),
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    "accuracy": results["overall_accuracy"],
                }

class BERTTokenClassificationModel(BaseTokenClassficationModel):
    def __init__(self, config):
        BaseTokenClassficationModel.__init__(self, config)
        config = AutoConfig.from_pretrained(self.model_name,num_labels=self.num_labels)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, config = config)
    
    def wipe_memory(self):
        self.model = None  
        self.optimizer = None 
        torch.cuda.empty_cache()

from .qa_utils import evaluate_metric
class BaseQuestionAnsweringModel:
    def __init__(self, config):
        self.model = nn.Module()
        self.model_name = config['model_name']
        self.vocab_size = config['vocab_size']
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.accelerator = Accelerator()

    def train(self, datasets, examples, **kwargs):
        save_dir = kwargs['save_dir']
        epochs = kwargs['epochs']
        lr = kwargs['lr']
        batch_size = kwargs['batch_size']
        self.optimizer = AdamW(self.model.parameters(), lr = lr)

        train_dataset, valid_dataset, test_dataset = datasets
        train_examples, valid_examples, test_examples = examples

        filepath = os.path.join(save_dir, 'model.pth')
        best_accuracy = 0 
        
        train_data_loader = train_dataset.remove_columns(["example_id", "offset_mapping"])
        train_data_loader.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])

        train_data_loader = torch.utils.data.DataLoader(train_data_loader, batch_size=batch_size)

        for epoch in range(epochs):
            accuracy = 0 
            loss = 0 
            self.model.train().to(self.device)
            all_start_logits = []
            all_end_logits = []
            for _, batch in enumerate(train_data_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs['loss']
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                all_start_logits.append(self.accelerator.gather(start_logits).detach().cpu().numpy())
                all_end_logits.append(self.accelerator.gather(end_logits).detach().cpu().numpy())
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss += loss / len(train_dataset)
                batch = None
            train_metrics = self.evaluate_dataset(train_dataset, train_examples, batch_size=batch_size)
            print(f"Epoch {epoch} Train Loss {loss:.4f} Train F1 {train_metrics['f1']:.4f}")
            
            self.model.eval().to(self.device)
            valid_metrics = self.evaluate_dataset(valid_dataset, valid_examples, batch_size=batch_size)
            print(f"Epoch {epoch} Valid Loss {valid_metrics['loss']:.4f} Valid F1 {valid_metrics['f1']:.4f}")

            val_accuracy = valid_metrics['f1']
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), filepath)
        
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        test_metrics = self.evaluate_dataset(test_dataset, test_examples, batch_size=batch_size)
        print(f"Epoch {epoch} Test Loss {test_metrics['loss']:.4f} Test F1 {test_metrics['f1']:.4f}")
        return {'f1':test_metrics['f1'], 'Exact Match':test_metrics['exact_match']}
        
    def evaluate_dataset(self, dataset, examples, batch_size = 8):
        loss = 0 
        all_start_logits = []
        all_end_logits = []
        data_loader = dataset.remove_columns(["example_id", "offset_mapping"])
        data_loader.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
        data_loader = torch.utils.data.DataLoader(data_loader, batch_size=batch_size)
        for _, batch in enumerate(data_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs['loss']
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            all_start_logits.append(self.accelerator.gather(start_logits).detach().cpu().numpy())
            all_end_logits.append(self.accelerator.gather(end_logits).detach().cpu().numpy())
            
            loss += loss / len(dataset)
            batch = None
        metric = evaluate_metric(dataset, examples, all_start_logits, all_end_logits)
        return {'loss':loss, 'f1':metric['f1']/100, 'exact_match':metric['exact_match']/100}

class BERTQuestionAnsweringModel(BaseQuestionAnsweringModel):
    def __init__(self, config):
        BaseQuestionAnsweringModel.__init__(self, config)
        config = AutoConfig.from_pretrained(self.model_name)
        self.model =  AutoModelForQuestionAnswering.from_pretrained(self.model_name, config = config)
    
    def wipe_memory(self):
        self.model = None  
        self.optimizer = None 
        torch.cuda.empty_cache()