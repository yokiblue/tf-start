import tensorflow as tf 
import pandas as pd 
import numpy as np 
import sys
import datetime
import matplotlib.pyplot as plt 
plt.style.use('ggplot') 


# function to model: y = a * x^2 + b * x + c

pool = np.random.rand(1000, 1).astype(np.float32)
np.random.shuffle(pool)
# ----sample size: 15%   15% for test|| 85% for training
sample = int(1000 * 0.15)
test_x = pool[0:sample]
train_x = pool[sample:]

#print('Testing data points: ' + str(test_x.shape))
#print('Training data points: ' + str(train_x.shape))

test_y = 2.0 * test_x ** 2 + 3.0 * test_x + 5
train_y = 2.0 * train_x ** 2 + 3.0 * train_x + 5

df = pd.DataFrame({'x':train_x[:,0], 'y':train_y[:,0]})
#print (df.head())
#print (df.describe())

#df.plot.scatter(x = 'x', y = 'y', figsize = (15, 5));
#plt.show()

hidden_size = 1
# shape: samples, the input neurons: None -> output ? means it can be of any shape
x = tf.placeholder(tf.float32, shape=[None, 1], name="01_x")
y = tf.placeholder(tf.float32, shape=[None, 1], name="01_y")

#print("shape of x and y: ")
#print(x.get_shape(), y.get_shape())

# create first hidden layer
W1 = tf.Variable(tf.truncated_normal([1, hidden_size], mean = 0.1, stddev = 0.1), name = "w1")
b1 = tf.Variable(tf.truncated_normal([hidden_size], mean = 0.1, stddev = 0.1), name = "b1")
h1 = tf.nn.relu(tf.matmul(x, W1) + b1, name = "h1")

#print ("shape of hidden layer: ")
#print (h1.get_shape())


# create output layer
W = tf.Variable(tf.truncated_normal([hidden_size, 1], mean = 0.1, stddev = 0.1), name = "w")
b = tf.Variable(tf.truncated_normal([1], mean = 0.1, stddev = 0.1), name = "b")

pred = tf.nn.relu(tf.matmul(h1,W) + b)

#print ("shape of output layer: ")
#print (pred.get_shape())

loss = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.GradientDescentOptimizer(0.09)
train = optimizer.minimize(loss)


# check model accuracy
correct_prediction = tf.equal(tf.round(pred), tf.round(y))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


init = tf.global_variables_initializer()
t = []
with tf.Session() as sess:
	sess.run(init)
	for step in range(250):
		train_data = {x: train_x, y: train_y}
		test_data = {x: test_x, y: test_y}

		train_loss, train_pred = sess.run([loss, train], feed_dict = train_data)
		if step%50 == 0:
			t.append((step, train_loss))
			train_acc = accuracy.eval(train_data)
			print ("Training loss at step %d: %f" % (step, train_loss))

	print ("Accuracy on the Training Set: ", accuracy.eval(train_data))
	print ("Accuracy on the Test Set: ", accuracy.eval(test_data))

	test_results = sess.run(pred, feed_dict = {x:test_x})
	df_final = pd.DataFrame({'test_x': test_x[:,0],
		'pred': test_results[:,0]})

	df_loss = pd.DataFrame(t, columns = ['step', 'train_loss'])


#fig, axes = plt.subplots (nrows = 1, ncols = 1, figsize = (15,5))
#df.plot.scatter(x = 'x', y = 'y', ax = axes , color = 'red')
#df_final.plot.scatter(x = 'test_x', y = 'pred', ax = axes, alpha = 0.3)
#axes.set_title('target vs pred', fontsize = 20)
#axes.set_ylabel('y', fontsize = 15)
#axes.set_xlabel('x', fontsize = 15)
#axes.legend(["target", "pred"], loc = 'best')

df_loss.set_index('step').plot(logy = True, figsize = (15,5))
plt.show()













	







