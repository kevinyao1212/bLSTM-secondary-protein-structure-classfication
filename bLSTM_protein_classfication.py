import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


# TODO: load dataset
data=np.load('../cullpdb+profile_6133_filtered.npy.gz')

# Parameters
training_epochs=10
batch_size=100

# Network Parameters
n_input = 22 # MNIST data input (img shape: 28*28)

# num of steps could vary, I randomly choose 50 amino acids in one step first
n_steps = 50 

# TODO: num of hidden layers and hidden layer units
# now using 128 firs
n_hidden = 128
n_hidden_1 = 128
n_hidden_2 = 128



n_classes = 9 # 8 secondary structure classes and a 'noseq' class

# input formatting
#all_x=data[:,0:22]
#all_y=data[:,22:31]
data.shape=(5534,700,57)
all_x=data[:,:,0:22]
all_y=data[:,:,22:31]

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

lstm_out = tf.placeholder("float", [None, n_classes] )

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'blstm_out': tf.Variable(tf.random_normal([2*n_hidden, n_classes])),

    #### TODO: should the input for h1 be the size of n_classes??????
    'h1': tf.Variable(tf.random_normal([n_classes, n_hidden_1])),

    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

## could be a benchmark, use a NN or just output bLSTM
## or maybe try different structure here like CNN 
##
##
def multilayer_NN(lstm_out, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(lstm_out, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    layer_2 = tf.add(tf.matmul(lstm_out, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def BiLSTM(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, int(n_steps), 0)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)

    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)




    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['blstm_out']) + biases['out']

lstm_out = BiLSTM(x, weights, biases)

pred = multilayer_NN(lstm_out, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations

    for epoch in range(training_epochs):
        total_batch=int(len(data)/batch_size)
        for i in  range(total_batch):
            #### load data ####
            batch_x=all_x[batch_size*i:batch_size*(i+1)]
            batch_y=all_y[batch_size*i:batch_size*(i+1)]
            # Reshape data to get 28 seq of 28 elements
            #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
    print ("Optimization Finished!")

    ####TODO，test set accuracy

    # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print "Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label})




