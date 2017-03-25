import tensorflow as tf
import numpy as np
import string
import sys
from datetime import datetime, timedelta
import importlib
import time
import pickle
import gzip
from tempfile import TemporaryFile

#f=gzip.open('cullpdb+profile_6133_filtered.npy.gz','rb')
#print(f)
X_in=np.load('cullpdb+profile_6133_filtered.npy.gz')

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

#config_name = "lstm_uni_20"
config = importlib.import_module("configurations.%s" % config_name)
opt = config.optimizer

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
elif opt == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
elif opt == "adagrad":
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
else:
    sys.exit("please choose either <rmsprop/adagrad/adadelta/nag> in configfile")

#f=gzip.open('../cb513+profile_split1.npy.gz','rb')
test=np.load('cb513+profile_split1.npy.gz')
test.shape=(514,700,57)
labelsTest = test[:,:,22:31]
a = np.arange(0,22)
b = np.arange(35,56)
c = np.hstack((a,b))
XTest = test[:,:,c]
count=0
for i in range(512):
    for j in range(700):
        if (XTest[i,j,21]==1):
            count+=1

init = tf.global_variables_initializer()

target=open("xyz.txt","w")

with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations

    for epoch in range(config.epochs):
        total_batch = int(len(X) / batch_size)

        if config_name == "multi_bLSTM":
            _fw_current_state = np.zeros((config.num_layers, 2, config.batch_size, config.N_LSTM_F))
            _bw_current_state = np.zeros((config.num_layers, 2, config.batch_size, config.N_LSTM_B))

        for i in range(total_batch):
            #### load data ####
            batch_x = X[batch_size * i:batch_size * (i + 1)]
            batch_y = labels[batch_size * i:batch_size * (i + 1)]
            # Reshape data to get 28 seq of 28 elements
            # batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)

            if config_name != "multi_bLSTM":
                _,loss=sess.run([optimizer,cost], feed_dict={config.x: batch_x, config.y: batch_y})
            else:
                _, loss, _fw_current_state, _bw_current_state  = sess.run(
                [optimizer, cost, config.fw_state, config.bw_state], 
                feed_dict={
                    config.x: batch_x, 
                    config.y: batch_y,
                    config.fw_state: _fw_current_state,
                    config.bw_state: _bw_current_state})

            #if i % display_step == 0:
            # Calculate batch accuracy
            #acc,loss = sess.run([accuracy,cost], feed_dict={config.x: batch_x, config.y: batch_y})
            # Calculate batch loss
            #loss = sess.run(cost, feed_dict={config.x: batch_x, config.y: batch_y})

            print("Iter " + str(i * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss))
        #Test accuracy
        acc = 0
        for i in range(int(514/config.batch_size)):
            batch_x = XTest[batch_size * i:batch_size * (i + 1)]
            batch_y = labelsTest[batch_size * i:batch_size * (i + 1)]
            prediction=tf.argmax(l_out,2)
            if config_name == "multi_bLSTM":
                res=prediction.eval(feed_dict={config.x:batch_x, config.y:batch_y, config.fw_state: _fw_current_state,
                        config.bw_state: _bw_current_state})
            else:
                res=prediction.eval(feed_dict={config.x:batch_x, config.y:batch_y})
                
            for j in range(config.batch_size):
                for k in range(config.seq_len):
                    if (batch_y[j,k,8]==1):
                        continue
                    if (np.argmax(batch_y[j,k,:])==res[j,k]):
                        acc+=1

        acc=acc/(512*700-count)
        print("test accuracy = "+str(acc))
        target.write(str(epoch)+" "+str(acc))
        target.write("\n")
        target.flush()
	    
    print("Optimization Finished!")
target.close()
