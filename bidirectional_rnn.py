import os
import sys
import warnings
import pickle
import datetime
import tensorflow as tf 
import pandas as pd
import traceback
import time 
import json
import numpy as np 
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras import Input 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.layers import Bidirectional 
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import BatchNormalization
warnings.filterwarnings("ignore")


####################  helper function #########################
def one_hot_encode(raw_y, num_classes):
  index = np.array(raw_y)
  class_cnt = num_classes #np.max(index) + 1 
  out = np.zeros((index.shape[0], class_cnt))
  out[np.arange(index.shape[0]), index] = 1
  return out 

def load_sample(fn, max_seq_len, word_dict, num_classes):
  text_df = pd.read_csv(fn)
  raw_y = []
  raw_x = []
  for i in range(len(text_df)):
    label = text_df['label'][i]
    raw_y.append(int(label))

    text = text_df['text'][i]
    text_len = len(text)
    x = np.zeros(max_seq_len, dtype = np.int32)
    if text_len <= max_seq_len:
      for i in range(text_len):
        x[i] = word_dict[text[i]]
    else:
      for i in range(text_len - max_seq_len, text_len):
        x[i - text_len + max_seq_len] = word_dict[text[i]]
    raw_x.append(x)

  all_x = np.array(raw_x)
  all_y = one_hot_encode(raw_y, num_classes)
  return all_x, all_y 

def batch_iter(x, y, batch_size = 16):
  data_len = len(x)
  num_batch = (data_len + batch_size - 1) // batch_size
  indices = np.random.permutation(np.arange(data_len))
  x_shuff = x[indices]
  y_shuff = y[indices]
  for i in range(num_batch):
    start_offset = i*batch_size 
    end_offset = min(start_offset + batch_size, data_len)
    yield i, num_batch, x_shuff[start_offset:end_offset], y_shuff[start_offset:end_offset]


######################### model start #####################
class RnnAttentionLayer(layers.Layer):
  def __init__(self, attention_size, drop_rate):
    super().__init__()
    self.attention_size = attention_size
    self.dropout = Dropout(drop_rate, name = "rnn_attention_dropout")

  def build(self, input_shape):
    self.attention_w = self.add_weight(name = "atten_w", shape = (input_shape[-1], self.attention_size), initializer = tf.random_uniform_initializer(), dtype = "float32", trainable = True)
    self.attention_u = self.add_weight(name = "atten_u", shape = (self.attention_size,), initializer = tf.random_uniform_initializer(), dtype = "float32", trainable = True)
    self.attention_b = self.add_weight(name = "atten_b", shape = (self.attention_size,), initializer = tf.constant_initializer(0.1), dtype = "float32", trainable = True)    
    super().build(input_shape)

  def call(self, inputs, training):
    x = tf.tanh(tf.add(tf.tensordot(inputs, self.attention_w, axes = 1), self.attention_b))
    x = tf.tensordot(x, self.attention_u, axes = 1)
    x = tf.nn.softmax(x)
    weight_out = tf.multiply(tf.expand_dims(x, -1), inputs)
    final_out = tf.reduce_sum(weight_out, axis = 1) 
    drop_out = self.dropout(final_out, training = training)
    return drop_out

class RnnLayer(layers.Layer):
  def __init__(self, rnn_size, drop_rate):
    super().__init__()
    fwd_lstm = LSTM(rnn_size, return_sequences = True, go_backwards= False, dropout = drop_rate, name = "fwd_lstm")
    bwd_lstm = LSTM(rnn_size, return_sequences = True, go_backwards = True, dropout = drop_rate, name = "bwd_lstm")
    self.bilstm = Bidirectional(merge_mode = "concat", layer = fwd_lstm, backward_layer = bwd_lstm, name = "bilstm")
    #self.bilstm = Bidirectional(LSTM(rnn_size, activation= "relu", return_sequences = True, dropout = drop_rate))

  def call(self, inputs, training):
    outputs = self.bilstm(inputs, training = training)
    return outputs
 
class Model(tf.keras.Model):
  def __init__(self, num_classes, drop_rate, vocab_size, embedding_size, rnn_size, attention_size):
    super().__init__()
    self.embedding_layer = Embedding(vocab_size, embedding_size, embeddings_initializer = "uniform", name = "embeding_0")
    self.rnn_layer = RnnLayer(rnn_size, drop_rate)
    self.attention_layer = RnnAttentionLayer(attention_size, drop_rate)
    self.dense_layer = Dense(num_classes, activation = "softmax", kernel_regularizer=keras.regularizers.l2(0.001), name = "dense_1")

  def call(self, input_x, training):
    x = self.embedding_layer(input_x)
    x = self.rnn_layer(x, training = training)
    x = self.attention_layer(x, training = training)
    x = self.dense_layer(x)
    return x

def train(xy_train, xy_val, num_classes, vocab_size, nbr_epoches, batch_size):
  uniq_cfg_name = datetime.datetime.now().strftime("%Y")
  model_prefix = os.path.join(os.getcwd(), "model")
  if not os.path.exists(model_prefix):
    print("create model dir: %s" % model_prefix)
    os.mkdir(model_prefix)

  model_path = os.path.join(model_prefix, uniq_cfg_name)
  model = Model(num_classes, drop_rate = 0.05, vocab_size = vocab_size, embedding_size = 256, rnn_size = 128, attention_size = 128)
  if os.path.exists(model_path):
    model.load_weights(model_path)
    print("load weight from: %s" % model_path)
  
  optimizer = tf.keras.optimizers.Adam(0.01)
  loss_fn = tf.keras.losses.CategoricalCrossentropy()

  loss_metric = tf.keras.metrics.Mean(name='train_loss')
  accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

  @tf.function 
  def train_step(input_x, input_y, training = True):
    with tf.GradientTape() as tape:
      raw_prob = model(input_x, training)
      #tf.print("raw_prob", raw_prob)
      pred_loss = loss_fn(input_y, raw_prob)
    gradients = tape.gradient(pred_loss, model.trainable_variables)
    if training:
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Update the metrics
    loss_metric.update_state(pred_loss)
    accuracy_metric.update_state(input_y, raw_prob)
    return raw_prob 

  for i in range(nbr_epoches):
    t0 = time.time()
    batch_train = batch_iter(xy_train[0], xy_train[1], batch_size = batch_size)
    loss_metric.reset_states()
    accuracy_metric.reset_states()

    for batch_no, batch_tot, data_x, data_y in batch_train:
      predict_prob = train_step(data_x, data_y, True)  
      #if batch_no % 10 == 0:
      #  print("[%d of %d]: loss: %0.3f acc %0.3f" % (batch_no, batch_tot, loss_metric.result(), accuracy_metric.result()))

    print("[train ep %d] [%s]: %0.3f  [%s]: %0.3f" %  (i, "loss", loss_metric.result() , "acc", accuracy_metric.result()))
    model.save_weights(model_path, overwrite=True)

    if (i + 1) % 5 == 0:
      loss_metric.reset_states()
      accuracy_metric.reset_states()
      batch_test = batch_iter(xy_val[0], xy_val[1], batch_size = batch_size)
      for _, _, data_x, data_y in batch_test:
        train_step(data_x, data_y, False)
      print("[***** ep %d] [%s]: %0.3f  [%s]: %0.3f" %  (i, "loss", loss_metric.result() , "acc", accuracy_metric.result()))

if __name__ == "__main__":
  try:
    cur_dir=os.getcwd()
    corps_meta_path = os.path.join(cur_dir, "corps_meta")
    corps_meta = pickle.load(open(corps_meta_path, "rb"))
    max_seq_len = min(64, corps_meta["max_seq_len"])
    num_classes = corps_meta["num_classes"] 
    word_dict = corps_meta["word_dict"] 
    index_dict = corps_meta["index_dict"]
    train_sample_path = os.path.join(cur_dir, "train.csv")
    test_sample_path = os.path.join(cur_dir, "test.csv")

    ### gen samples ###
    train_x, train_y = load_sample(train_sample_path, max_seq_len, word_dict, num_classes)
    test_x, test_y = load_sample(test_sample_path, max_seq_len, word_dict, num_classes)
    key, freq = np.unique(np.argmax(train_y, axis = 1), return_counts = True)
    train([train_x, train_y], [test_x, test_y], num_classes, len(word_dict), nbr_epoches = 100, batch_size = 256)
  except:
    traceback.print_exc()