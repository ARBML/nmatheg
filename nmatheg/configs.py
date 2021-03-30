def create_configs(config, data_config):
    dataset_name = config['dataset']['dataset_name']
    train_config = {'epochs':int(config['train']['epochs']),
                    'save_dir':config['train']['save_dir']}

    model_config = {'model_name':config['model']['model_name'],
                    'vocab_size':int(config['tokenization']['vocab_size']),
                    'num_labels':int(data_config[dataset_name]['num_labels'])}
    return train_config, model_config