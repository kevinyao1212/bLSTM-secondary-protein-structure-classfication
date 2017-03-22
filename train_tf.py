import tensorflow as tf
import numpy as np
import string
import sys
from datetime import datetime, timedelta
import importlib
import time
import pickle
import gzip

f=gzip.open('../cullpdb+profile_6133_filtered.npy.gz','rb')
X_in=np.load(f)

X = np.reshape(X_in,(5534,700,57))
del X_in
X = X[:,:,:]
labels = X[:,:,22:31]
#mask = X[:,:,30] * -1 + 1

a = np.arange(0,22)
b = np.arange(35,56)
c = np.hstack((a,b))
X = X[:,:,c]

learning_rate = 0.001
display_step = 50

np.random.seed(1)

if len(sys.argv) != 2:
    sys.exit("Usage: python train.py <config_name>")

config_name = sys.argv[1]
test_name=config_name+"_test"

#config_name = "lstm_uni_20"
config = importlib.import_module("configurations.%s" % config_name)
test=importlib.import_module("configurations.%s" % test_name)
opt = config.optimizer
print("Using configurations: '%s'" % config_name)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_id = "%s-%s" % (config_name, timestamp)
metadata_path = "metadata/dump_%s" % experiment_id

print("Experiment id: %s" % experiment_id)



#def main():
num_epochs = config.epochs
batch_size = config.batch_size

l_out = config.build_model(config.x,config.weights,config.biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l_out, labels=config.y))


# Evaluate model
correct_pred = tf.equal(tf.argmax(l_out, 2), tf.argmax(config.y, 2))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

if opt == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
# elif optimizer == "adadelta":
#
# elif optimizer == "adagrad":
#
# elif optimizer == "nag":

else:
    sys.exit("please choose either <rmsprop/adagrad/adadelta/nag> in configfile")

#import data
#X_train, X_valid, y_train, y_valid, mask_train, mask_valid, num_seq_train = data.get_train()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations

    for epoch in range(config.epochs):
        total_batch = int(len(X) / batch_size)
        for i in range(total_batch):
            if (i==1):
                break
            #### load data ####
            batch_x = X[batch_size * i:batch_size * (i + 1)]
            batch_y = labels[batch_size * i:batch_size * (i + 1)]
            # Reshape data to get 28 seq of 28 elements
            # batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={config.x: batch_x, config.y: batch_y})

            #if i % display_step == 0:
            # Calculate batch accuracy
            acc,loss = sess.run([accuracy,cost], feed_dict={config.x: batch_x, config.y: batch_y})
            # Calculate batch loss
            #loss = sess.run(cost, feed_dict={config.x: batch_x, config.y: batch_y})
            print("Iter " + str(i * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
	    
    print("Optimization Finished!")


    #Test accuracy
    l_out = test.build_model(test.x,test.weights,test.biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l_out, labels=test.y))

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(l_out, 2), tf.argmax(test.y, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    f=gzip.open('../cb513+profile_split1.npy.gz','rb')
    test=np.load(f)
    test.shape=(514,700,57)
    labels = test[:,:,22:30]
    a = np.arange(0,22)
    b = np.arange(35,56)
    c = np.hstack((a,b))
    X = test[:,:,c]
    count=0
    for i in range(514):
        for j in range(700):
            if (X[i,j,21]==1):
                count+=1
    print(count)
    acc = 0
    for i in range(int(514/test.batch_size)):
        batch_x = X[batch_size * i:batch_size * (i + 1)]
        batch_y = labels[batch_size * i:batch_size * (i + 1)]
        acc+=sess.run(accuracy,feed_dict={test.x:batch_x, test.y:batch_y})
        #acc=accuracy.eval({config.x:X, config.y:labels})
    acc/=int(514/test.batch_size)
    acc=(514*700*acc)/(514*700-count)
    print("test accuracy = "+str(acc))
