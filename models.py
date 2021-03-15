import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GRU, Embedding, Dense, Input, Dropout, Bidirectional

def create_model(vocab_size, num_classes = 1):
    model = Sequential()
    model.add(Embedding(vocab_size, 128))
    model.add(Bidirectional(GRU(units = 256, return_sequences = True)))
    model.add(Bidirectional(GRU(units = 256)))
    model.add(Dense(num_classes, activation = 'sigmoid'))
    return model