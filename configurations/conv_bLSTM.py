import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


#validate_every = 40
start_saving_at = 0
save_every = 20
#write_every_batch = 10

epochs = 5
batch_size = 64
N_CONV_A = 4
F_CONV_A = 7
N_L1 = 200
N_LSTM_F = 400
N_LSTM_B = 400
N_L2 = 200
n_inputs = 43
n_classes = 9
seq_len = 700
optimizer = "rmsprop"
lambda_reg = 0.0001
cut_grad = 20

# tf Graph input
x = tf.placeholder("float", [None, seq_len, n_inputs])
y = tf.placeholder("float", [None, seq_len, n_classes])

# Define weights
weights = {
    'h1': tf.Variable(tf.random_normal([seq_len*N_CONV_A, N_L1])),
    'h2': tf.Variable(tf.random_normal([N_LSTM_F*2, N_L2])),
    'out': tf.Variable(tf.random_normal([N_L2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([N_L1])),
    'b2': tf.Variable(tf.random_normal([N_L2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def build_model(x, weights, biases):
    x = tf.transpose(x, [0, 2, 1])
    x = tf.reshape(x, [-1, seq_len, 1])

    y_conv_1 = tf.layers.conv1d(x, N_CONV_A, F_CONV_A, padding='same', 
                                activation=tf.nn.relu)

    batch_conv_1 = tf.layers.batch_normalization(y_conv_1)

    batch_flat = tf.reshape(batch_conv_1,[-1, seq_len*N_CONV_A])

    layer_1 = tf.add(tf.matmul(batch_flat, weights['h1']), biases['b1'])
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

    output_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    output_layer=tf.reshape(output_layer,[batch_size,seq_len,n_classes])

    return output_layer

