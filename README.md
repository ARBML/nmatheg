
 <p align="center"> 
 <img src = "https://raw.githubusercontent.com/ARBML/nmatheg/master/nmatheg_logo.PNG" width = "200px"/>
 </p>


# nmatheg

nmatheg `نماذج` an easy straregy for training Arabic NLP models on huggingface datasets. Just specifiy the name of the dataset, preprocessing, tokenization and the training procedure in the config file to train an nlp model for that task. 

## install 

```pip install nmatheg```

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

### Main Sections 

- `dataset` describe the dataset and the task type. Currently we only support classification 
- `preprocessing` a set of cleaning functions mainly uses our library [tnkeeh](https://github.com/ARBML/tnkeeh). 
- `tokenization` descrbies the tokenizer used for encoding the dataset. It uses our library [tkseem](https://github.com/ARBML/tkseem). 
- `train` the training parameters like number of epochs and batch size. 

## Usage 
```python
import nmatheg as nm
strategy = nm.TrainStrategy('config.ini')
strategy.start()
```

## Datasets 
We are supporting huggingface datasets for Arabic. You can find the supported datasets [here](https://github.com/ARBML/nmatheg/blob/main/nmatheg/datasets.ini). 

## Tasks 

Currently only supporting classification tasks using bidirectional GRUs. We are aplanning to support more complicated mechanisms like BERT fine-tuning. 

## Demo 
Check this [colab notebook](https://colab.research.google.com/github/ARBML/nmatheg/blob/main/demo.ipynb) for a quick demo. 