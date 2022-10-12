import configparser
def create_default_config(batch_size = 64, epochs = 5, lr = 5e-5, runs = 10, max_tokens = 64, 
                          max_train_samples = -1, preprocessing = {}, ckpt = 'ckpts'):
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

    for arg in preprocessing:
        config['preprocessing'][arg] = preprocessing[arg]

    config['tokenization'] = {
        'max_tokens' : max_tokens,
        'tok_save_path': 'ckpts', 
        'max_train_samples': max_train_samples
    }

    config['log'] = {'print_every':10}

    config['train'] = {
        'save_dir' : ckpt,
        'epochs' : epochs,
        'batch_size' : batch_size,
        'lr': lr, 
        'runs': runs 
    }
    return config 