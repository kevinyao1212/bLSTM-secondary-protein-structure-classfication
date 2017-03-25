import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

epochs = 400
batch_size = 64
N_L1 = 128
N_LSTM_F = 50
N_LSTM_B = 50
N_L2 = 128
n_inputs = 43
n_classes = 9
seq_len = 700
num_layers = 3
optimizer = "rmsprop"

# tf Graph input
x = tf.placeholder("float", [None, seq_len, n_inputs])
y = tf.placeholder("float", [None, seq_len, n_classes])

fw_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, N_LSTM_F])
bw_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, N_LSTM_B])

fw_state_per_layer_list = tf.unstack(fw_state, axis=0)
bw_state_per_layer_list = tf.unstack(bw_state, axis=0)


fw_tuple_state = tuple(
    [tf.contrib.rnn.core_rnn_cell.LSTMStateTuple(fw_state_per_layer_list[idx][0], fw_state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)

bw_tuple_state = tuple(
    [tf.contrib.rnn.core_rnn_cell.LSTMStateTuple(bw_state_per_layer_list[idx][0], bw_state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)
# Define weights
weights = {
    'h1': tf.Variable(tf.random_normal([n_inputs, N_L1])),
    'h2': tf.Variable(tf.random_normal([N_LSTM_F*2, N_L2])),
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
    lstm_forward = rnn.BasicLSTMCell(N_LSTM_F)
    lstm_forward = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([lstm_forward] * num_layers, state_is_tuple=True)

    lstm_backward = rnn.BasicLSTMCell(N_LSTM_B)
    lstm_backward = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([lstm_backward] * num_layers, state_is_tuple=True)

    #state = tf.zeros([batch_size, lstm_forward.state_size])

    #lstm_output, state = lstm_forward(layer_1, state)
    lstm_output, fw_state, bw_state = rnn.static_bidirectional_rnn(lstm_forward, lstm_backward, layer_1, initial_state_fw=fw_tuple_state, 
        initial_state_bw=bw_tuple_state, dtype=tf.float32)

    lstm_output = tf.reshape(lstm_output,[batch_size*seq_len,2*N_LSTM_F])
    layer_2 = tf.nn.dropout(lstm_output, keep_prob = 0.5)

    layer_2 = tf.add(tf.matmul(layer_2, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    output_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    output_layer=tf.reshape(output_layer,[batch_size,seq_len,n_classes])

    return output_layer

