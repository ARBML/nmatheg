import nmatheg as nm
strategy = nm.TrainStrategy(
    datasets = 'ARGEN_title_generation', 
    models   = 'UBC-NLP/AraT5-base', 
    runs = 10,
    lr = 5e-5,
    epochs = 20,
    batch_size = 4,
    max_tokens = 128,
)
output = strategy.start()