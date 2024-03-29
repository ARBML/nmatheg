# Creating a class to pull the words from the columns and create them into sentences
from datasets import Dataset, DatasetDict

def aggregate_tokens(dataset, config, data_config, max_len = 128):
    new_dataset = {}
    token_col = data_config['text']
    tag_col = data_config['label']

    for split in dataset:
        sent_labels = []
        sent_label = []
        sentence = []
        sentences = []
        
        for i, item in enumerate(dataset[split]):
            token, label = item[token_col], item[tag_col]
            sent_label.append(label)
            sentence.append(token)
            if len(sentence) == max_len:
                sentences.append(sentence) 
                sent_labels.append(sent_label)
                sentence = []
                sent_label = []
        new_dataset[split] = Dataset.from_dict({token_col:sentences, tag_col:sent_labels}) 
    return DatasetDict(new_dataset) 

# https://github.com/huggingface/transformers/blob/44f5b260fe7a69cbd82be91b58c62a2879d530fa/examples/pytorch/token-classification/run_ner_no_trainer.py#L353
def tokenize_and_align_labels(dataset, tokenizer, data_config, model_type = 'transformer', max_len = 128):

    token_col = data_config['text']
    tag_col = data_config['label']

    if 'transformer' in model_type:
        tokenized_inputs = tokenizer(
            dataset[token_col],
            max_length=max_len,
            padding='max_length',
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(dataset[tag_col]):
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
    else:
        labels = []
        input_ids = []
        for i, label in enumerate(dataset[tag_col]):
            word_ids = []
            tokens = []
            for j, word in enumerate(dataset[token_col][i]):
                token_ids = tokenizer._encode_word(word)
                for token_id in token_ids:
                    tokens.append(token_id)
                    word_ids.append(j)
                if len(tokens) > max_len:
                    break
            
            while len(tokens) < max_len:
                tokens.append(0)
                word_ids.append(None)
            else:
                tokens = tokens[:max_len]
                word_ids = word_ids[:max_len]
            
            input_ids.append(tokens)
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
        dataset["labels"] = labels
        dataset["input_ids"] = input_ids
        return dataset