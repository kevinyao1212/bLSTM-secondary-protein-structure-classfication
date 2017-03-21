import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


#validate_every = 40
start_saving_at = 0
save_every = 20
#write_every_batch = 10

epochs = 400
batch_size = 64
N_L1 = 200
N_LSTM_F = 400
N_LSTM_B = 400
N_L2 = 200
n_inputs = 42
n_classes = 8
seq_len = 700
optimizer = "rmsprop"
lambda_reg = 0.0001
cut_grad = 20

# tf Graph input
x = tf.placeholder("float", [None, seq_len, n_inputs])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'h1': tf.Variable(tf.random_normal([n_inputs, N_L1])),
    'h2': tf.Variable(tf.random_normal([N_LSTM_F, N_L2])),
    'out': tf.Variable(tf.random_normal([N_L2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([N_L1])),
    'b2': tf.Variable(tf.random_normal([N_L2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def build_model(x, weights, biases):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    lstm_forward = rnn.BasicLSTMCell(N_LSTM_F)
    #lstm_backward = rnn.BasicLSTMCell(N_LSTM_B)

    state = tf.zeros([batch_size, lstm_forward.state_size])

    lstm_output, state = lstm_forward(layer_1, state)

    layer_2 = tf.nn.dropout(lstm_output, keep_prob = 0.5)

    layer_2 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    output_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

    return output_layer