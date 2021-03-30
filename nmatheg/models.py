from transformers import AutoModelForSequenceClassification, AutoConfig
import os
import time
from tqdm import tqdm
import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import torch.nn as nn

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

class BaseModel:
    def __init__(self, config):
        self.model = nn.Module()
        self.num_labels = config['num_labels']
        self.model_name = config['model_name']
        self.vocab_size = config['vocab_size']

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def train(self, datasets, epochs = 30, save_dir = '.'):
        train_dataset, valid_dataset, test_dataset = datasets 
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
                self.optimizer.zero_grad()

                labels = batch['labels'].cpu() 
                preds = outputs['logits'].argmax(-1).cpu() 
                accuracy += accuracy_score(labels, preds) /len(train_dataset)
                loss += loss / len(train_dataset)
            
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
        results = self.evaluate_dataset(test_dataset)
        print(f"Test Loss {results['loss']:.4f} Test Accuracy {results['accuracy']:.4f}")
    
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
        return {'loss':loss, 'accuracy':accuracy}

class SimpleClassificationModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.model = BiRNN(self.vocab_size, self.num_labels)
        self.model.to(self.device)    
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

class BERTClassificationModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        config = AutoConfig.from_pretrained(self.model_name,num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config = config)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)