import configparser
def create_default_config(batch_size = 4, epochs = 5, lr = 5e-5):
    config = configparser.ConfigParser()

    config['preprocessing'] = {
        'segment' : False,
        'remove_special_chars' : False,
        'remove_english' : False,
        'normalize' : False,
        'remove_diacritics' : False,
        'excluded_chars' : [],
        'remove_tatweel' : False,
        'remove_html_elements' : False,
        'remove_links' : False,
        'remove_twitter_meta' : False,
        'remove_long_words' : False,
        'remove_repeated_chars' : False,
    }

    config['tokenization'] = {
        'tokenizer_name' : 'WordTokenizer',
        'vocab_size' : 10000,
        'max_tokens' : 128,
    }

    config['log'] = {'print_every':10}

    config['train'] = {
        'save_dir' : '.',
        'epochs' : epochs,
        'batch_size' : batch_size,
        'lr': lr
    }
    return config 

def create_configs(config, data_config):
    dataset_name = config['dataset']['dataset_name']
    train_config = {'epochs':int(config['train']['epochs']),
                    'save_dir':config['train']['save_dir'],
                    'batch_size':int(config['train']['batch_size']),
                    'lr':float(config['train']['lr'])}

    model_config = {'model_name':config['model']['model_name'],
                    'vocab_size':int(config['tokenization']['vocab_size']),
                    'num_labels':int(data_config[dataset_name]['num_labels'])}
    return train_config, model_config