
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
dataset_name = ajgt_twitter_ar

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
vocab_size = 1000
max_tokens = 128

[model]
model_name = rnn

[log]
print_every = 10

[train]
save_dir = .
epochs = 10
batch_size = 256 
```

### Main Sections 

- `dataset` describe the dataset and the task type. Currently we only support classification 
- `preprocessing` a set of cleaning functions mainly uses our library [tnkeeh](https://github.com/ARBML/tnkeeh). 
- `tokenization` descrbies the tokenizer used for encoding the dataset. It uses our library [tkseem](https://github.com/ARBML/tkseem). 
- `train` the training parameters like number of epochs and batch size. 

## Usage 

### Config Files
```python
import nmatheg as nm
strategy = nm.TrainStrategy('config.ini')
strategy.start()
```
### Benchmarking on multiple datasets and models 
```python
import nmatheg as nm
strategy = nm.TrainStrategy(
    datasets = 'arsentd_lev,arcd,caner', 
    models   = 'qarib/bert-base-qarib,aubmindlab/bert-base-arabertv01'
)
strategy.start()
```

## Datasets 
We are supporting huggingface datasets for Arabic. You can find the supported datasets [here](https://github.com/ARBML/nmatheg/blob/main/nmatheg/datasets.ini). 

| Dataset | Description |
| --- | --- |
| [ajgt_twitter_ar](https://huggingface.co/datasets/ajgt_twitter_ar) | Arabic Jordanian General Tweets (AJGT) Corpus consisted of 1,800 tweets annotated as positive and negative. Modern Standard Arabic (MSA) or Jordanian dialect. |
| [metrec](https://huggingface.co/datasets/metrec) | The dataset contains the verses and their corresponding meter classes. Meter classes are represented as numbers from 0 to 13. The dataset can be highly useful for further research in order to improve the field of Arabic poems’ meter classification. The train dataset contains 47,124 records and the test dataset contains 8,316 records. |
|[labr](https://huggingface.co/datasets/labr) |This dataset contains over 63,000 book reviews in Arabic. It is the largest sentiment analysis dataset for Arabic to-date. The book reviews were harvested from the website Goodreads during the month or March 2013. Each book review comes with the goodreads review id, the user id, the book id, the rating (1 to 5) and the text of the review. |
|[ar_res_reviews](https://huggingface.co/datasets/ar_res_reviews)|Dataset of 8364 restaurant reviews from qaym.com in Arabic for sentiment analysis|
|[arsentd_lev](https://huggingface.co/datasets/arsentd_lev)|The Arabic Sentiment Twitter Dataset for Levantine dialect (ArSenTD-LEV) contains 4,000 tweets written in Arabic and equally retrieved from Jordan, Lebanon, Palestine and Syria.|
|[oclar](https://huggingface.co/datasets/oclar)|The researchers of OCLAR Marwan et al. (2019), they gathered Arabic costumer reviews Zomato [website](https://www.zomato.com/lebanon) on wide scope of domain, including restaurants, hotels, hospitals, local shops, etc. The corpus finally contains 3916 reviews in 5-rating scale. For this research purpose, the positive class considers rating stars from 5 to 3 of 3465 reviews, and the negative class is represented from values of 1 and 2 of about 451 texts.|
|[emotone_ar](https://huggingface.co/datasets/emotone_ar)|Dataset of 10,065 tweets in Arabic for Emotion detection in Arabic text|
|[hard](https://huggingface.co/datasets/hard)|This dataset contains 93,700 hotel reviews in Arabic language.The hotel reviews were collected from Booking.com website during June/July 2016.The reviews are expressed in Modern Standard Arabic as well as dialectal Arabic.The following table summarize some tatistics on the HARD Dataset.|
|[caner](https://huggingface.co/datasets/caner)|The Classical Arabic Named Entity Recognition corpus is a new corpus of tagged data that can be useful for handling the issues in recognition of Arabic named entities.|
|[arcd](https://huggingface.co/datasets/arcd)|Arabic Reading Comprehension Dataset (ARCD) composed of 1,395 questions posed by crowdworkers on Wikipedia articles.|

## Tasks 

Currently we support text classification, named entity recognition and question answering. 

## Demo 
Check this [colab notebook](https://colab.research.google.com/github/ARBML/nmatheg/blob/main/demo.ipynb) for a quick demo. 