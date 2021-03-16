import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GRU, Embedding, Dense, Input, Dropout, Bidirectional
import os
import time

class ClassificationModel():
    def __init__(self, vocab_size, num_classes, ckpt_dir):
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        self.valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        self.valid_accuracy = tf.keras.metrics.Mean(name='valid_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.model = self.create_classification_model()
        
        self.checkpoint = tf.train.Checkpoint(
                optimizer=self.optimizer,
                model=self.model
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.checkpoint,
            ckpt_dir,
            max_to_keep=1,
            checkpoint_name='ckpt',
        )
        
    def create_classification_model(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 128))
        model.add(Bidirectional(GRU(units = 256, return_sequences = True)))
        model.add(Bidirectional(GRU(units = 256)))
        model.add(Dense(self.num_classes, activation = 'tanh'))
        return model
    

    def loss_function(self, real, pred):
        loss_ = self.loss_object(real, pred)
        return tf.reduce_mean(loss_)

    def accuracy_function(self, real, pred):
        accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=-1), tf.int32))
        return tf.reduce_mean(tf.cast(accuracies, tf.float32))


    def train_step(self, inp, tar):
        with tf.GradientTape() as tape:
            predictions = self.model(inp)
            loss = self.loss_function(tar, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(self.accuracy_function(tar, predictions))

    # @tf.function(input_signature=train_step_signature)
    def valid_step(self, inp, tar):
        predictions = self.model(inp)
        loss = self.loss_function(tar, predictions)
        
        self.valid_loss(loss)
        self.valid_accuracy(self.accuracy_function(tar, predictions))

    # @tf.function(input_signature=train_step_signature)
    def test_step(self, inp, tar):
        predictions = self.model(inp)
        loss = self.loss_function(tar, predictions)
        
        self.test_loss(loss)
        self.test_accuracy(self.accuracy_function(tar, predictions))

    def train(self, datasets, epochs = 30, verbose = 0):
        train_dataset, valid_dataset, test_dataset = datasets
        best_score = 10
        for epoch in range(epochs):
            start = time.time()
            
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            self.valid_loss.reset_states()
            self.valid_accuracy.reset_states()
            
            # inp -> portuguese, tar -> english
            for (batch, (inp, tar)) in enumerate(train_dataset):
                self.train_step(inp, tar)
                
                if batch % 500 == 0 and verbose:
                    print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))

            for (batch, (inp, tar)) in enumerate(valid_dataset):
                self.valid_step(inp, tar)
                
                
                print ('Epoch {} Train Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                            self.train_loss.result(), 
                                                            self.train_accuracy.result()))
                
                print ('Epoch {} Valid Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                            self.valid_loss.result(), 
                                                            self.valid_accuracy.result()))
                
                print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            if  self.valid_loss.result() < best_score:
                best_score = self.valid_loss.result()
                ckpt_save_path = os.path.basename(os.path.normpath(self.ckpt_manager.save()))
                
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
        
        # restore best model
        self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
        _, test_score = self.evaluate_test(test_dataset)

        print('test score is:',test_score)

    def evaluate_test(self, test_dataset):
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        for (batch, (inp, tar)) in enumerate(test_dataset):
            self.test_step(inp, tar)

        return self.test_loss.result().numpy(), self.test_accuracy.result().numpy()
