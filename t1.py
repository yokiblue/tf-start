import tensorflow as tf
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')



#print ('Python version ' + sys.version)
#print ('Tensorflow version ' + tf.VERSION)
#print ('Pandas version' + pd.__version__)


#a = tf.constant(5.0)
#b = tf.constant(6.0)
#c = a * b

#with tf.Session() as sess:
#	print (sess.run(c))
#	print (c.eval())

#print (type(a))
#print (type(b))
#print (type(c))

#a = tf.placeholder(tf.float32, name = "var_a")
#b = tf.placeholder(tf.float32, name = "var_b")
#c = tf.multiply(a, b)

#with tf.Session() as sess: 
#	print (sess.run(c, feed_dict = {a:[7.0], b:[8.0]}))

#df = pd.DataFrame({'a': [2,4,6,8],
#	'b':[2,2,2,2]})

#a = tf.placeholder(tf.int32, name = "var_a")
#b = tf.placeholder(tf.int32, name = "var_b")

#c = tf.multiply(a,b)

#with tf.Session() as sess:
#	print (sess.run(c, feed_dict = {a: df['a'].tolist(), b:df['b'].tolist()}))


#radius = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
#area = [3.14159, 12.56636, 28.27431, 50.26544, 78.53975, 113.09724]
#plt.plot(radius, area)


train_x = np.random.rand(100).astype(np.float32)
train_y = 0.1 * train_x + 0.3 

df = pd.DataFrame({'x':train_x, 'y':train_y})
#print (df.head())
#print (df.describe())
#----plot training data
#df.plot.scatter(x = 'x', y = 'y', figsize = (15,5))
#plt.show()

test_x = np.random.rand(100).astype(np.float32)

x = tf.placeholder(tf.float32, name = "01_x")
y = tf.placeholder(tf.float32, name = "01_y")

W = tf.Variable(np.random.rand())
b = tf.Variable(np.random.rand())
pred = tf.multiply(W, x) + b

loss = tf.reduce_mean(tf.square(pred - y))

optimizer = tf.train.GradientDescentOptimizer(0.7)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for step in range(200): 
		train_data = {x:train_x, y:train_y}
		sess.run(train,feed_dict = train_data)
		if step > 180: 
			print (step, sess.run(W), sess.run(b))
	print ("Training completed: ", "W = ", sess.run(W), "b = ", sess.run(b))

	test_results = sess.run(pred, feed_dict = {x:test_x})

	df_final = pd.DataFrame({'test_x':test_x, 'pred':test_results})



fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15,5))
df.plot.scatter(x = 'x', y = 'y', ax = axes, color = 'red')
df_final.plot.scatter(x = 'test_x', y = 'pred', ax = axes, alpha = 0.3)

axes.set_title('target vs pred', fontsize = 20)
axes.set_ylabel('y', fontsize = 15)
axes.set_xlabel('x', fontsize = 15)
plt.show()





















