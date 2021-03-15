
import os 
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GRU, Embedding, Dense, Input, Dropout, Bidirectional
import tkseem as tk
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import tnkeeh as tn
import argparse
import logging 
import sys
from datasets import load_dataset
from utils import *
import models
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

MAX_TOKENS = int(config['tokenization']['max_tokens'])
tokenizer_name = config['tokenization']['tokenizer_name']
tokenizer = get_tokenizer(tokenizer_name)
vocab_size = int(config['tokenization']['vocab_size'])

dataset_name = config['dataset']['dataset_name']

train_dir = config['train']['dir']
epochs = int(config['train']['epochs'])

def prepare_data(data_name):
  write_dataset(dataset_name, config)

  train_text = open('data/train_data.txt', 'r').read().splitlines()
  train_lbls = [int(lbl) for lbl in open('data/train_lbls.txt', 'r').read().splitlines()]
  valid_text = open('data/valid_data.txt', 'r').read().splitlines()
  valid_lbls = [int(lbl) for lbl in open('data/valid_lbls.txt', 'r').read().splitlines()]
  test_text = open('data/test_data.txt', 'r').read().splitlines()
  test_lbls = [int(lbl) for lbl in open('data/test_lbls.txt', 'r').read().splitlines()]
  assert len(train_text) == len(train_lbls)
  assert len(test_text) == len(test_lbls)
  assert len(valid_text) == len(valid_lbls)
  return train_text, valid_text, test_text, train_lbls, valid_lbls, test_lbls

def tokenize_data(dataset_name, tokenizer, vocab_size = 10000):
  train_text, valid_text, test_text, train_lbls, valid_lbls, test_lbls = prepare_data(dataset_name)
  tokenizer = tokenizer(vocab_size = vocab_size)
  tokenizer.train('data/train_data.txt')
  train_data = tokenizer.encode_sentences(train_text, out_length=MAX_TOKENS)
  valid_data = tokenizer.encode_sentences(valid_text, out_length=MAX_TOKENS)
  test_data = tokenizer.encode_sentences(test_text, out_length=MAX_TOKENS)
  return tokenizer, (train_data, train_lbls), (valid_data, valid_lbls), (test_data, test_lbls)

def create_dataset(train_data, valid_data, test_data, batch_size = 256, buffer_size = 50000):
  train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

  valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
  valid_dataset = valid_dataset.batch(batch_size)

  test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
  test_dataset = test_dataset.batch(batch_size)
  return train_dataset, valid_dataset, test_dataset


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False,
              reduction = 'none')

def loss_function(real, pred):
  loss_ = loss_object(real, pred)
  return tf.reduce_mean(loss_)

def accuracy_function(real, pred):
  result = tf.equal(tf.squeeze(real),tf.squeeze(tf.cast(tf.round(pred), tf.int32))) 
  return tf.reduce_mean( tf.cast(result, tf.float32))

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.Mean(name='valid_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = loss_function(tar, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  train_loss(loss)
  train_accuracy(accuracy_function(tar, predictions))

# @tf.function(input_signature=train_step_signature)
def valid_step(inp, tar):
  predictions = model(inp)
  loss = loss_function(tar, predictions)
  
  valid_loss(loss)
  valid_accuracy(accuracy_function(tar, predictions))

# @tf.function(input_signature=train_step_signature)
def test_step(inp, tar):
  predictions = model(inp)
  loss = loss_function(tar, predictions)
  
  test_loss(loss)
  test_accuracy(accuracy_function(tar, predictions))

def train(epochs = 30, verbose = 0):
  best_score = 10
  for epoch in range(epochs):
    start = time.time()
    
    train_loss.reset_states()
    train_accuracy.reset_states()

    valid_loss.reset_states()
    valid_accuracy.reset_states()
    
    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_dataset):
      train_step(inp, tar)
      
      if batch % 500 == 0 and verbose:
        print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        
      

    for (batch, (inp, tar)) in enumerate(valid_dataset):
      valid_step(inp, tar)
      
      
    print ('Epoch {} Train Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                  train_loss.result(), 
                                                  train_accuracy.result()))
    
    print ('Epoch {} Valid Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                  valid_loss.result(), 
                                                  valid_accuracy.result()))
    
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    if  valid_loss.result() < best_score:
      best_score = valid_loss.result()
      ckpt_save_path = os.path.basename(os.path.normpath(ckpt_manager.save()))
      
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

def evaluate_test(test_dataset):
  test_loss.reset_states()
  test_accuracy.reset_states()
  for (batch, (inp, tar)) in enumerate(test_dataset):
      test_step(inp, tar)

  return test_loss.result().numpy(), test_accuracy.result().numpy()

results = {}

BATCH_SIZE = 256

checkpoint_dir = f'{train_dir}/ckpts/'

tokenizer, train_data, valid_data, test_data = tokenize_data(dataset_name, tokenizer, vocab_size = vocab_size)
train_dataset, valid_dataset, test_dataset = create_dataset(train_data, valid_data, test_data, batch_size = BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
model = models.create_model(vocab_size)

# create checkpoint object
checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model
)
ckpt_manager = tf.train.CheckpointManager(
    checkpoint,
    f'{checkpoint_dir}/vocab_size_{vocab_size}/{tokenizer_name}/',
    max_to_keep=1,
    checkpoint_name='ckpt',
)

train(epochs = epochs)

# restore best model
checkpoint.restore(ckpt_manager.latest_checkpoint)
_, test_score = evaluate_test(test_dataset)

results['test_score'] = test_score
print('results on test score is:',results)
