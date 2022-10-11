import nmatheg as nm
strategy = nm.TrainStrategy(
    datasets = 'ajgt_twitter_ar,caner,xnli', 
    models   = 'birnn', 
    tokenizers = 'BPE,MaT-BPE,Seg-BPE',
    vocab_sizes = '250,500,1000,5000,10000',
    runs = 10,
    lr = 1e-3,
    epochs = 20,
    batch_size = 128,
    max_tokens = 128,
    mode = 'pretrain'
)
output = strategy.start()