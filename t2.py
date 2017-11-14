import tensorflow as tf
import pandas as pd 
import numpy as np 
import sys 
import math
import matplotlib.pyplot as plt

train_x = np.random.rand(100).astype(np.float32)
train_y = 0.1 * train_x + 0.3

x = tf.placeholder(tf.float32, name = "01_x")
y = tf.placeholder(tf.float32, name = "01_y")

W = tf.Variable(np.random.rand())
b = tf.Variable(np.random.rand())
pred = W * train_x + b

loss = tf.reduce_mean(tf.square(pred - train_y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# check accuracy of model
correct_prediction = tf.equal(tf.round(pred), tf.round(train_y))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


t = []
init = tf.global_variables_initializer()
with tf.Session() as sess: 
	sess.run(init)
	for step in range(200):
		train_data = {x:train_x, y:train_y}
		sess.run(train, feed_dict = train_data)
		t.append((step, sess.run(loss, feed_dict = train_data)))
	print ("Accuracy on the Training Set: ", accuracy.eval({x: train_x, y:train_y}))


#df_error = pd.DataFrame(t, columns = ['step', 'error'])
#df_error.plot(x = 'step', y = 'error')

pd.DataFrame(t, columns=['step', 'error']).set_index('step').plot(figsize=(20,5));

plt.show()












