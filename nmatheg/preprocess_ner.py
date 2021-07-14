# Creating a class to pull the words from the columns and create them into sentences
import torch 
from datasets import Dataset, DatasetDict

def process_dataset(dataset):
    new_dataset = {}
    
    for split in dataset:
        sent_labels = []
        sent_label = []
        sentence = []
        sentences = []
        
        for i, item in enumerate(dataset[split]):
            token, label = item['token'], item['ner_tag']
            sent_label.append(label)
            sentence.append(token)
            if len(sentence) == 512:
                sentences.append((' ').join(sentence)) 
                sent_labels.append(sent_label)
                sentence = []
                sent_label = []
        new_dataset[split] = Dataset.from_dict({'token':sentences, 'ner_tag':sent_labels}) 
    return DatasetDict(new_dataset) 