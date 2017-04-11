import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


epochs = 5
batch_size = 64
N_CONV_A = 16
N_CONV_B = 16
N_CONV_C = 16
F_CONV_A = 3
F_CONV_B = 5
F_CONV_C = 7
N_L1 = 200
N_LSTM_F = 250
N_LSTM_B = 250
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
    'h1': tf.Variable(tf.random_normal([n_inputs*N_CONV_A, N_L1])),
    'h2': tf.Variable(tf.random_normal([N_LSTM_F*2, N_L2])),
    'out': tf.Variable(tf.random_normal([N_L2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([N_L1])),
    'b2': tf.Variable(tf.random_normal([N_L2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def build_model(x, weights, biases):
    #input_layer = tf.transpose(x, [0, 2, 1])
    #x = tf.reshape(x, [-1, seq_len, 1])

    y_conv_1 = tf.layers.conv1d(x, N_CONV_A, F_CONV_A, padding='same', 
                                activation=tf.nn.relu)

    batch_conv_1 = tf.layers.batch_normalization(y_conv_1)

    #batch_flat = tf.reshape(batch_conv_1,[-1, n_inputs*N_CONV_A])

    y_conv_2 = tf.layers.conv1d(x, N_CONV_B, F_CONV_B, padding='same', 
                                activation=tf.nn.relu)

    batch_conv_2 = tf.layers.batch_normalization(y_conv_2)

    y_conv_3 = tf.layers.conv1d(x, N_CONV_C, F_CONV_C, padding='same', 
                                activation=tf.nn.relu)

    batch_conv_3 = tf.layers.batch_normalization(y_conv_3)

    concat_layer_1 = tf.concat([batch_conv_1, batch_conv_2, batch_conv_3], 2)
    #concat_layer_1 = tf.transpose(concat_layer_1, [0, 2, 1])

    concat_layer_2 = tf.concat([x, concat_layer_1], 2)

    reshape_layer_1 = tf.reshape(concat_layer_2, [batch_size*seq_len, n_inputs+48])

    dense_layer_1 = tf.layers.dense(reshape_layer_1, N_L1, activation=tf.nn.relu, use_bias=True)
    
    batch_dense_1 = tf.layers.batch_normalization(dense_layer_1)
    
    layer_1=tf.split(batch_dense_1,seq_len,0)
    lstm_forward = rnn.BasicLSTMCell(N_LSTM_F)
    lstm_backward = rnn.BasicLSTMCell(N_LSTM_B)

    #state = tf.zeros([batch_size, lstm_forward.state_size])

    #lstm_output, state = lstm_forward(layer_1, state)
    lstm_output, _, _ = rnn.static_bidirectional_rnn(lstm_forward, lstm_backward, 
            layer_1, dtype=tf.float32)

    lstm_output=tf.reshape(lstm_output,[batch_size*seq_len,2*N_LSTM_F])
    layer_2 = tf.nn.dropout(lstm_output, keep_prob = 0.5)

    dense_layer_2 = tf.layers.dense(layer_2, N_L2, activation=tf.nn.relu, use_bias=True)

    output_layer = tf.add(tf.matmul(dense_layer_2, weights['out']), biases['out'])
    output_layer=tf.reshape(output_layer,[batch_size,seq_len,n_classes])

    return output_layer

