import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


epochs = 50
batch_size = 64
N_L1 = 200
N_GRU_F = 250
#N_LSTM_B = 400
N_L2 = 200
n_inputs = 43
n_classes = 9
seq_len = 700
optimizer = "rmsprop"

# tf Graph input
x = tf.placeholder("float", [None, seq_len, n_inputs])
y = tf.placeholder("float", [None, seq_len, n_classes])

# Define weights
weights = {
    'h1': tf.Variable(tf.random_normal([n_inputs, N_L1])),
    'h2': tf.Variable(tf.random_normal([N_GRU_F, N_L2])),
    'out': tf.Variable(tf.random_normal([N_L2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([N_L1])),
    'b2': tf.Variable(tf.random_normal([N_L2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def build_model(x, weights, biases):

    x=tf.reshape(x,[-1,n_inputs])
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    #layer_1=tf.reshape(layer_1,[batch_size,seq_len,N_L1])
    layer_1=tf.split(layer_1,seq_len,0)
    GRU_forward = rnn.GRUCell(N_GRU_F)
    #lstm_backward = rnn.BasicLSTMCell(N_LSTM_B)

    #state = tf.zeros([batch_size, lstm_forward.state_size])

    #lstm_output, state = lstm_forward(layer_1, state)
    GRU_output, _ = rnn.static_rnn(GRU_forward, layer_1, dtype=tf.float32)

    GRU_output = tf.reshape(GRU_output,[batch_size*seq_len,N_GRU_F])
    layer_2 = tf.nn.dropout(GRU_output, keep_prob = 0.5)

    layer_2 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    output_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    output_layer=tf.reshape(output_layer,[batch_size,seq_len,n_classes])

    return output_layer

