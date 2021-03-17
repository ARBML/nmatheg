
 <p align="center"> 
 <img src = "https://raw.githubusercontent.com/ARBML/nmatheg/master/nmatheg_logo.PNG" width = "200px"/>
 </p>

# nmatheg

Nmatheg `نماذج` an easy straregy for training Arabic NLP models on huggingface datasets. Just specifiy the name of the dataset, preprocessing, tokenization and the training procedure in the config file to train an nlp model for that task. 

## Configuration

Setup a config file for the training strategy. 

``` ini
[dataset]
dataset_name = ajgt_twitter_ar
task = classification 

[preprocessing]
segment = False
remove_special_chars = False
remove_english = False
normalize = False
remove_diacritics = False
excluded_chars = []
remove_tatweel = False
remove_html_elements = False
remove_links = False 
remove_twitter_meta = False
remove_long_words = False
remove_repeated_chars = False

[tokenization]
tokenizer_name = WordTokenizer
vocab_size = 10000
max_tokens = 128

[train]
dir = .
epochs = 10
batch_size = 256
```

## Usage 
```python
import nmatheg as nm
strategy = nm.TrainStrategy('config.ini')
strategy.start()
```

## Datasets 
We are supporting huggingface datasets for Arabic. You can find the supported datasets [here](https://github.com/ARBML/nmatheg/blob/main/nmatheg/datasets.ini). 

## Models 

- Classification Models 

## Demo 
Check this [colab notebook](https://colab.research.google.com/github/ARBML/nmatheg/blob/main/demo.ipynb) for a quick demo. 