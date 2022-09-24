import nmatheg as nm
strategy = nm.TrainStrategy(
    datasets = 'caner', 
    models   = 'birnn', 
    tokenizers = 'bpe',
    vocab_sizes = '1000',
    runs = 1,
    lr = 1e-4,
    epochs = 50,
    batch_size = 128,
    max_tokens = 128,
)
output = strategy.start()