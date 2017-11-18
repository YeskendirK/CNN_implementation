import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 9])
#yn = tf.placeholder(tf.float32, shape = [None, 1])

#Remove digit=9 from training data
positions = np.nonzero(mnist.train.labels[:,9]==1)
positions = positions[0]
new_train_images = np.delete(mnist.train.images, positions, axis = 0)
new_train_labels = np.delete(mnist.train.labels, positions, axis = 0)
new_train_labels = np.delete(new_train_labels, [9], axis = 1)

#Remove digit = 9 from validation data
positions = np.nonzero(mnist.validation.labels[:,9]==1)
positions = positions[0]
new_valid_images = np.delete(mnist.validation.images, positions, axis = 0)
new_valid_labels = np.delete(mnist.validation.labels, positions, axis = 0)
new_valid_labels = np.delete(new_valid_labels, [9], axis = 1)

#Remove digit = 9 from testing data
positions = np.nonzero(mnist.test.labels[:,9]==1)
positions = positions[0]
new_test_images = np.delete(mnist.test.images, positions, axis = 0)
new_test_labels = np.delete(mnist.test.labels, positions, axis = 0)
new_test_labels = np.delete(new_test_labels, [9], axis = 1)

# Convolutional layer
x_image = tf.reshape(x, [-1,28,28,1])
W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, 30], stddev=0.1))
b_conv = tf.Variable(tf.constant(0.1, shape=[30]))
h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID')
h_relu = tf.nn.relu(h_conv + b_conv)
h_pool = tf.nn.max_pool(h_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully-connected layer
W_fc1 = tf.Variable(tf.truncated_normal([12 * 12 * 30, 500], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[500]))
h_pool_flat = tf.reshape(h_pool, [-1, 12*12*30])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

# Output layer
W_fc2 = tf.Variable(tf.truncated_normal([500, 9], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[9]))
y_hat=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Train and Evaluate the Model
cross_entropy = - tf.reduce_sum(y_*tf.log(y_hat))
train_step = tf.train.AdamOptimizer(2e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
print("=================================")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
for j in range(5):
    for i in range(550):
        positions_ = np.random.choice(new_train_images.shape[0], 100, replace = False)
        batch_x = new_train_images[positions_,:]
        batch_y_ = new_train_labels[positions_,:]
        #print(np.size(batch_y_))
        train_step.run(feed_dict={x: batch_x, y_: batch_y_})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_: batch_y_})
            val_accuracy = accuracy.eval(feed_dict={x: new_valid_images, y_:new_valid_labels})
            print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))

print("|===============================|")
test_accuracy = accuracy.eval(feed_dict=\
    {x: new_test_images, y_:new_test_labels})
print("test accuracy=%.4f"%(test_accuracy))

#Add one more neuron
W_fc3 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b_fc3 = tf.Variable(tf.constant(0.1, shape=[10]))
y_hat3=tf.nn.softmax(tf.matmul(h_fc1, W_fc3) + b_fc3)

#Train only final layer
y = tf.placeholder(tf.float32, shape = [None, 10])
cross_entropy3 = - tf.reduce_sum(y*tf.log(y_hat3))
train_step3 = tf.train.AdamOptimizer(2e-3).minimize(cross_entropy3, var_list = [W_fc3, b_fc3])
correct_prediction3 = tf.equal(tf.argmax(y_hat3,1), tf.argmax(y,1))
accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))

sess.run(tf.initialize_all_variables())
print("=================================")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
for j in range(5):
    for i in range(550):
        batch = mnist.train.next_batch(100)
        train_step3.run(feed_dict={x: batch[0], y: batch[1]})
        if i%50 == 49:
            train_accuracy = accuracy3.eval(feed_dict={x:batch[0], y: batch[1]})
            val_accuracy = accuracy3.eval(feed_dict=\
                {x: mnist.validation.images, y:mnist.validation.labels})
            print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))
print("|===============================|")
test_accuracy = accuracy3.eval(feed_dict=\
    {x: mnist.test.images, y:mnist.test.labels})
print("test accuracy=%.4f"%(test_accuracy))
