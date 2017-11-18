import tensorflow as tf
import math
import numpy as np
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

#Parameter Specification
nh = 20
lr = 0.05
num_input = 2
num_output = 1
num_samples_0 = 30000
num_samples_1 = 25000
num_samples = num_samples_0 + num_samples_1
num_valid_0 = 2500
num_valid_1 = 2500
num_valid = num_valid_0 + num_valid_1
num_test_0 = 5000
num_test_1 = 5000
num_test = num_test_0 + num_test_1
mu, sigma = 0, 1

# Symbolic variables
x_ = tf.placeholder(tf.float32, shape=[None,2])
y_ = tf.placeholder(tf.float32, shape=[None,1])

#Generating Training data
r0 = np.random.normal(mu, sigma, num_samples_0)
t0 = np.random.uniform(0,2*np.pi,num_samples_0)
r0 = np.array(r0)
t0 = np.array(t0)
r1 = np.random.normal(mu, sigma, num_samples_1)
t1 = np.random.uniform(0,2*np.pi,num_samples_1)
r1 = np.array(r1)
t1 = np.array(t1)
x1_0 = r0*np.cos(t0)
x2_0 = r0*np.sin(t0)
x1_1 = (r1+5)*np.cos(t1)
x2_1 = (r1+5)*np.sin(t1)
y_0 = np.zeros(num_samples_0)
y_1 = np.ones(num_samples_1)
x_train = np.empty([num_samples, 2])
y_train = np.empty([num_samples, 1])
x_train[0:num_samples_0,0:1] = x1_0.reshape(num_samples_0,1)
x_train[0:num_samples_0,1:2] = x2_0.reshape(num_samples_0,1)
y_train[0:num_samples_0,0:1] = y_0.reshape(num_samples_0,1)
x_train[num_samples_0:num_samples,0:1] = x1_1.reshape(num_samples_1,1)
x_train[num_samples_0:num_samples,1:2] = x2_1.reshape(num_samples_1,1)
y_train[num_samples_0:num_samples,0:1] = y_1.reshape(num_samples_1,1)

# Generating Validation data
r0 = np.random.normal(mu, sigma, num_valid_0)
t0 = np.random.uniform(0,2*np.pi,num_valid_0)
r0 = np.array(r0)
t0 = np.array(t0)
r1 = np.random.normal(mu, sigma, num_valid_1)
t1 = np.random.uniform(0,2*np.pi,num_valid_1)
r1 = np.array(r1)
t1 = np.array(t1)
x1_0 = r0*np.cos(t0)
x2_0 = r0*np.sin(t0)
x1_1 = (r1+5)*np.cos(t1)
x2_1 = (r1+5)*np.sin(t1)
y_0 = np.zeros(num_valid_0)
y_1 = np.ones(num_valid_1)
x_valid = np.empty([num_valid, 2])
y_valid = np.empty([num_valid, 1])
x_valid[0:num_valid_0,0:1] = x1_0.reshape(num_valid_0,1)
x_valid[0:num_valid_0,1:2] = x2_0.reshape(num_valid_0,1)
y_valid[0:num_valid_0,0:1] = y_0.reshape(num_valid_0,1)
x_valid[num_valid_0:num_valid,0:1] = x1_1.reshape(num_valid_1,1)
x_valid[num_valid_0:num_valid,1:2] = x2_1.reshape(num_valid_1,1)
y_valid[num_valid_0:num_valid,0:1] = y_1.reshape(num_valid_1,1)

#Generating Test data
r0 = np.random.normal(mu, sigma, num_test_0)
t0 = np.random.uniform(0,2*np.pi,num_test_0)
r0 = np.array(r0)
t0 = np.array(t0)
r1 = np.random.normal(mu, sigma, num_test_1)
t1 = np.random.uniform(0,2*np.pi,num_test_1)
r1 = np.array(r1)
t1 = np.array(t1)
x1_0 = r0*np.cos(t0)
x2_0 = r0*np.sin(t0)
x1_1 = (r1+5)*np.cos(t1)
x2_1 = (r1+5)*np.sin(t1)
y_0 = np.zeros(num_test_0)
y_1 = np.ones(num_test_1)
x_test = np.empty([num_test, 2])
y_test = np.empty([num_test, 1])
x_test[0:num_test_0,0:1] = x1_0.reshape(num_test_0,1)
x_test[0:num_test_0,1:2] = x2_0.reshape(num_test_0,1)
y_test[0:num_test_0,0:1] = y_0.reshape(num_test_0,1)
x_test[num_test_0:num_test,0:1] = x1_1.reshape(num_test_1,1)
x_test[num_test_0:num_test,1:2] = x2_1.reshape(num_test_1,1)
y_test[num_test_0:num_test,0:1] = y_1.reshape(num_test_1,1)


#Weights and biases initialization
w1 = tf.Variable(tf.truncated_normal([num_input, nh], stddev = 0.1))
w2 = tf.Variable(tf.truncated_normal([nh, num_output], stddev = 0.1))
b1 = tf.Variable(tf.truncated_normal([nh], stddev = 0.1))
b2 = tf.Variable(tf.truncated_normal([num_output], stddev = 0.1))

#Activation Setting
h = tf.nn.relu(tf.matmul(x_, w1)+b1)
yhat = tf.nn.sigmoid(tf.matmul(h, w2)+b2)

# Train and Evaluate the Model
cost = tf.reduce_mean((y_-yhat)**2)
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)

correct_prediction = tf.equal(tf.cast(tf.greater(yhat,0.5), tf.float32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#next batch
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#--------------------
# Run optimization
#--------------------
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

print("=================================")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
for j in range(3):
    for i in range(550):
        batch = next_batch(100, x_train, y_train)
        train_step.run(feed_dict={x_: batch[0], y_: batch[1]})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x_:batch[0], y_: batch[1]})
            val_accuracy = accuracy.eval(feed_dict=\
                {x_: x_valid, y_:y_valid})
            print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))
print("|===============================|")
test_accuracy = accuracy.eval(feed_dict=\
    {x_: x_test, y_: y_test})
print("test accuracy=%.4f"%(test_accuracy))
