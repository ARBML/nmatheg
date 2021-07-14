# Creating a class to pull the words from the columns and create them into sentences
import torch 
from datasets import Dataset, DatasetDict
#TODO this will ony work for caner
def aggregate_tokens(dataset):
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
                sentences.append(sentence) 
                sent_labels.append(sent_label)
                sentence = []
                sent_label = []
        new_dataset[split] = Dataset.from_dict({'token':sentences, 'ner_tag':sent_labels}) 
    return DatasetDict(new_dataset) 

# https://github.com/huggingface/transformers/blob/44f5b260fe7a69cbd82be91b58c62a2879d530fa/examples/pytorch/token-classification/run_ner_no_trainer.py#L353
def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples['token'],
        max_length=128,
        padding='max_length',
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples['ner_tag']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if True else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs