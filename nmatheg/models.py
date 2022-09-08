from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup)
from evaluate import load
import random 
import torch.nn.functional as F
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
from .qa_utils import evaluate_metric

class BiRNN(nn.Module):
    def __init__(self, vocab_size, num_labels, hidden_dim = 128):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.bigru1 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True)
        self.bigru2 = nn.GRU(2*hidden_dim, hidden_dim, bidirectional=True)
        self.bigru3 = nn.GRU(2*hidden_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
    def forward(self, 
                input_ids,
                labels):

        embedded = self.embedding(input_ids)        
        out,h = self.bigru1(embedded)
        out,h = self.bigru2(out)
        out,h = self.bigru3(out)
        logits = self.fc(out[:,0,:])
        loss = self.compute_loss(logits, labels)
        return {'loss':loss,
                'logits':logits} 
    
    def compute_loss(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
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
        filepath = os.path.join(save_dir, 'model.pth')
        best_accuracy = 0 
        for epoch in range(epochs):
            accuracy = 0 
            loss = 0 
            self.model.train().to(self.device)
            for _, batch in enumerate(train_dataset):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs['loss']
                loss.backward()
                self.optimizer.step()
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
        # self.optimizer = AdamW(self.model.parameters(), lr = 5e-5)

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

class BiRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size, num_labels, hidden_dim = 128):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.bigru1 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True)
        self.bigru2 = nn.GRU(2*hidden_dim, hidden_dim, bidirectional=True)
        self.bigru3 = nn.GRU(2*hidden_dim, hidden_dim//2, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
    def forward(self, 
                input_ids,
                labels):

        embedded = self.embedding(input_ids)        
        out,h = self.bigru1(embedded)
        out,h = self.bigru2(out)
        out,h = self.bigru3(out)
        logits = self.fc(out)
        loss = self.compute_loss(logits, labels)
        return {'loss':loss,
                'logits':logits} 
    
    def compute_loss(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss

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
            print(f"Epoch {epoch} Train Loss {train_metrics['loss']:.4f} Train F1 {train_metrics['f1']:.4f}")
            
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

class SimpleTokenClassificationModel(BaseTokenClassficationModel):
    def __init__(self, config):
        BaseTokenClassficationModel.__init__(self, config)
        self.model = BiRNNForTokenClassification(self.vocab_size, self.num_labels)
        self.model.to(self.device)  
        # self.optimizer = AdamW(self.model.parameters(), lr = 5e-5)

    def wipe_memory(self):
        self.model = None  
        self.optimizer = None 
        torch.cuda.empty_cache()

class BERTTokenClassificationModel(BaseTokenClassficationModel):
    def __init__(self, config):
        BaseTokenClassficationModel.__init__(self, config)
        config = AutoConfig.from_pretrained(self.model_name,num_labels=self.num_labels)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, config = config)
    
    def wipe_memory(self):
        self.model = None  
        self.optimizer = None 
        torch.cuda.empty_cache()

class BaseQuestionAnsweringModel:
    def __init__(self, config):
        self.model = nn.Module()
        self.model_name = config['model_name']
        self.vocab_size = config['vocab_size']
        self.num_labels = config['num_labels'] 
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
        
        # train_data_loader = train_dataset.remove_columns(["example_id", "offset_mapping"])
        # train_data_loader.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])

        # train_data_loader = torch.utils.data.DataLoader(train_data_loader, batch_size=batch_size)

        for epoch in range(epochs):
            accuracy = 0 
            loss = 0 
            # self.model.train().to(self.device)
            all_start_logits = []
            all_end_logits = []
            for _, batch in enumerate(train_dataset):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                val = batch['input_ids']
                val[val==-100] = 0 
                outputs = self.model(**batch)
                loss = outputs['loss']
                start_logits = outputs['start_logits']
                end_logits = outputs['end_logits']

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
        for _, batch in enumerate(dataset):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            val = batch['input_ids']
            val[val==-100] = 0 
            outputs = self.model(**batch)
            loss = outputs['loss']
            start_logits = outputs['start_logits']
            end_logits = outputs['end_logits']

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

class BiRNNForQuestionAnswering(nn.Module):
    def __init__(self, vocab_size, num_labels = 2, hidden_dim = 128):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.bigru1 = nn.GRU(hidden_dim, hidden_dim, bidirectional=True)
        self.bigru2 = nn.GRU(2*hidden_dim, hidden_dim, bidirectional=True)
        self.bigru3 = nn.GRU(2*hidden_dim, hidden_dim//2, bidirectional=True)
        self.qa_outputs = nn.Linear(hidden_dim, num_labels)
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
    def forward(self, 
                input_ids,
                start_positions = None,
                end_positions = None):

        embedded = self.embedding(input_ids)        
        out,h = self.bigru1(embedded)
        out,h = self.bigru2(out)
        out,h = self.bigru3(out)
        logits = self.qa_outputs(out)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        loss = self.compute_loss(start_logits, end_logits, start_positions, end_positions)
        return {'loss':loss,
                'logits':logits,
                'start_logits':start_logits,
                'end_logits':end_logits} 
    
    def compute_loss(self, start_logits, end_logits, start_positions, end_positions):
        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss

class SimpleQuestionAnsweringModel(BaseQuestionAnsweringModel):
    def __init__(self, config):
        BaseQuestionAnsweringModel.__init__(self, config)
        self.model = BiRNNForQuestionAnswering(self.vocab_size, self.num_labels)
        self.model.to(self.device)  
        # self.optimizer = AdamW(self.model.parameters(), lr = 5e-5)

    def wipe_memory(self):
        self.model = None  
        self.optimizer = None 
        torch.cuda.empty_cache()


class BaseMachineTranslationModel:
    def __init__(self, config, tokenizer = None):
        self.model = nn.Module()
        self.model_name = config['model_name']
        self.vocab_size = config['vocab_size']
        self.num_labels = config['num_labels']
        self.tokenizer = tokenizer 
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def train(self, datasets, examples,  **kwargs):
        save_dir = kwargs['save_dir']
        epochs = kwargs['epochs']
        lr = kwargs['lr']
        batch_size = kwargs['batch_size']

        self.optimizer = AdamW(self.model.parameters(), lr = lr)
        self.metric = load("sacrebleu")
        train_dataset, valid_dataset, test_dataset = datasets

        filepath = os.path.join(save_dir, 'model.pth')
        best_accuracy = 0 
        
        for epoch in range(epochs):
            loss = 0 
            self.model.train().to(self.device)
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
            print(f"Epoch {epoch} Train Loss {loss:.4f} Train BLEU {train_metrics['bleu']:.4f}")
            
            self.model.eval().to(self.device)
            valid_metrics = self.evaluate_dataset(valid_dataset)
            print(f"Epoch {epoch} Valid Loss {valid_metrics['loss']:.4f} Valid BLEU {valid_metrics['bleu']:.4f}")

            val_accuracy = valid_metrics['bleu']
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), filepath)
        
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        test_metrics = self.evaluate_dataset(test_dataset)
        print(f"Epoch {epoch} Test Loss {test_metrics['loss']:.4f} Test BLEU {test_metrics['bleu']:.4f}")
        return {'bleu':test_metrics['bleu']}
        
    def evaluate_dataset(self, dataset):
        loss = 0 
        bleu_score = 0
        for _, batch in enumerate(dataset):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if self.model_name == "seq2seq":
              outputs = self.model(**batch, mode ="generate")
              generated_tokens = outputs['outputs']
            else:
              outputs = self.model(**batch)
              generated_tokens = self.model.generate(batch['input_ids'])
            labels = batch['labels']
            loss = outputs['loss']
            metric = self.compute_metrics(generated_tokens.cpu(), labels.cpu())
            bleu_score += metric['bleu'] / len(dataset)
            loss += loss / len(dataset)
        
        return {'loss':loss, 'bleu':bleu_score}

    def compute_metrics(self, preds, labels):
        if isinstance(preds, tuple):
            preds = preds[0]
        
        if 'T5' in self.model_name:
          decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

          # Replace -100 in the labels as we can't decode them.
          labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
          decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
          result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
          result = {"bleu": result["score"]}
          result = {k: round(v, 4) for k, v in result.items()}
          return result
        else:
          
          preds = self.get_lists(preds)
          labels  = self.get_lists(labels)
          
          decoded_preds = self.tokenizer.decode_sentences(preds)

          decoded_labels = self.tokenizer.decode_sentences(labels)
          decoded_labels = [[stmt] for stmt in decoded_labels]

          result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
          result = {"bleu": result["score"]}
          result = {k: round(v, 4) for k, v in result.items()}
          return result
    
    def get_lists(self, inputs):
        inputs = inputs.cpu().detach().numpy().astype(int).tolist()
        output = []
        for input in inputs:
          current_item =[]
          for item in input:
            if item == self.tokenizer.eos_idx:
              break
            else:
              current_item.append(item)
          output.append(current_item)
        return output



class T5MachineTranslationModel(BaseMachineTranslationModel):
    def __init__(self, config, tokenizer = None):
        BaseMachineTranslationModel.__init__(self, config, tokenizer = tokenizer)
        config = AutoConfig.from_pretrained(self.model_name)
        self.model =  AutoModelForSeq2SeqLM.from_pretrained(self.model_name, config = config)
    
    def wipe_memory(self):
        self.model = None  
        self.optimizer = None 
        torch.cuda.empty_cache()

#https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb#scrollTo=dCK3LIN25n_S
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers)
                
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.embedding(src)
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.embedding(input)
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Seq2SeqMachineTranslation(nn.Module):
    def __init__(self, vocab_size = 500, tokenizer = None):
        super().__init__()
        ENC_EMB_DIM = 128
        DEC_EMB_DIM = 128
        HID_DIM = 2084
        N_LAYERS = 1
        INPUT_DIM = vocab_size
        OUTPUT_DIM = vocab_size
        self.vocab_size = vocab_size
        self.encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS)
        self.decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = tokenizer 
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, input_ids, labels, teacher_forcing_ratio = 0.0, mode = "train"):
        src = torch.transpose(input_ids, 0, 1)
        trg = torch.transpose(labels, 0, 1)

        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens          
        input = torch.tensor([self.tokenizer.sos_idx]*batch_size).to(self.device)
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            
            if mode == "train" and teacher_force:
              input = trg[t]
            else:
              input = top1
            
            outputs[t] = output
        
        
        loss = self.compute_loss(outputs, trg)
        return {'loss':loss,
              'outputs':torch.transpose(outputs.argmax(-1), 0, 1)
              } 

    def compute_loss(self, output, trg):
        loss_fct = nn.CrossEntropyLoss(ignore_index = self.tokenizer.pad_idx)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].reshape(-1)
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = loss_fct(output, trg)
        return loss


class SimpleMachineTranslationModel(BaseMachineTranslationModel):
    def __init__(self, config, tokenizer = None):
        BaseMachineTranslationModel.__init__(self, config, tokenizer = tokenizer)
        self.model = Seq2SeqMachineTranslation(vocab_size = self.vocab_size, tokenizer = tokenizer)
        self.model.to(self.device)  
        # self.optimizer = AdamW(self.model.parameters(), lr = 5e-5)

    def wipe_memory(self):
        self.model = None  
        self.optimizer = None 
        torch.cuda.empty_cache()