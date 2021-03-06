import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

epochs = 50
batch_size = 64
N_L1 = 200
N_LSTM_F = 250
N_LSTM_B = 250
N_L2 = 256
N_L3 = 128
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
    'h2': tf.Variable(tf.random_normal([N_LSTM_F*2, N_L2])),
    'h3': tf.Variable(tf.random_normal([N_L2, N_L3])),
    'out': tf.Variable(tf.random_normal([N_L2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([N_L1])),
    'b2': tf.Variable(tf.random_normal([N_L2])),
    'b3': tf.Variable(tf.random_normal([N_L3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def build_model(x, weights, biases):

    x=tf.reshape(x,[-1,n_inputs])
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    #layer_1=tf.reshape(layer_1,[batch_size,seq_len,N_L1])
    layer_1=tf.split(layer_1,seq_len,0)
    lstm_forward = rnn.BasicLSTMCell(N_LSTM_F)
    lstm_backward = rnn.BasicLSTMCell(N_LSTM_B)

    #state = tf.zeros([batch_size, lstm_forward.state_size])

    #lstm_output, state = lstm_forward(layer_1, state)
    lstm_output, _, _ = rnn.static_bidirectional_rnn(lstm_forward, lstm_backward, 
            layer_1, dtype=tf.float32)

    lstm_output=tf.reshape(lstm_output,[batch_size*seq_len,2*N_LSTM_F])
    layer_2 = tf.nn.dropout(lstm_output, keep_prob = 0.5)

    layer_2 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    output_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    output_layer=tf.reshape(output_layer,[batch_size,seq_len,n_classes])

    return output_layer

