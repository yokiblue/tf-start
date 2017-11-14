import tensorflow as tf 
import pandas as pd 
import numpy as np 
import sys
import datetime
import matplotlib.pyplot as plt 
plt.style.use('ggplot')


# Function y = a * x^4 + b    training 70% validation 15% test 15%

pool = np.random.rand(1000, 1).astype(np.float32)
np.random.shuffle(pool)
sample = int(1000 * 0.15)
test_x = pool[0 : sample]
valid_x = pool[sample : sample * 2]
train_x = pool[sample*2 :]

print ('Testing data points: ' + str(test_x.shape))
print ('Validation data points: ' + str(valid_x.shape))
print ('Training data points: ' + str(train_x.shape))

test_y = 2.0 * test_x ** 4 + 5
valid_y = 2.0 * valid_x ** 4 + 5
train_y = 2.0 * train_x ** 4 + 5

df = pd.DataFrame({'x': train_x[:,0],'y': train_y[:,0]})
#print (df.head())
#print (df.describe())

#df.plot.scatter(x = 'x', y = 'y', figsize = (15,5))
#plt.show()

def add_layer(inputs, in_size, out_size, activation_function = None):
	Weights = tf.Variable(tf.truncated_normal([in_size, out_size], mean = 0.1, stddev = 0.1))
	biases = tf.Variable(tf.truncated_normal([out_size], mean = 0.1, stddev = 0.1))

	pred = tf.matmul(inputs, Weights) + biases
	if activation_function is None: 
		outputs = pred
	else: 
		outputs = activation_function(pred)
	return outputs


hidden_size = 100
x = tf.placeholder(tf.float32, shape = [None, 1], name = "01_x")
y = tf.placeholder(tf.float32, shape = [None, 1], name = "01_y")

h1 = add_layer(x, 1, hidden_size, tf.nn.relu)
h2 = add_layer(h1, hidden_size, hidden_size, tf.nn.relu)

print ("Shape of hidden layers: ", h1.get_shape(), h1.get_shape())

pred = add_layer(h2, hidden_size, 1)
print ("Shape of output layer: ", pred.get_shape())

loss = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.GradientDescentOptimizer(0.005)
train = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.round(pred), tf.round(y))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# best validation accuracy seen so far
best_valid_acc = 0.0

# interation-number for last improvement to validation accuracy
last_improvement = 0

# stop optimization if no improvement found in this may iteration
require_improvement = 1500


init = tf.global_variables_initializer()
t = []
with tf.Session() as sess:
	sess.run(init)
	for step in range(8000):
		train_data = {x: train_x, y: train_y}
		valid_data = {x: valid_x, y: valid_y}
		test_data = {x: test_x, y: test_y}

		train_loss, train_pred = sess.run([loss, train], feed_dict = train_data)

		if step % 100 == 0: 
			valid_loss = sess.run(loss, feed_dict = valid_data)
			t.append((step, train_loss, valid_loss))

			train_acc = accuracy.eval(train_data)
			valid_acc = accuracy.eval(valid_data)

			# if validation accuracy is an improvement over best-known
			if valid_acc > best_valid_acc: 
				best_valid_acc = valid_acc
				last_improvement = step
				improved_str = '*'
			else:
				improved_str = ''

			print ("Training loss at step %d: %f %s" % (step, train_loss, improved_str))
			print ("Validation %f" % (valid_loss))

			# if current step - last improvement step > 1500, stop
			if step - last_improvement > require_improvement: 
				print ("No improvement found in a while, stopping optimization")
				break

	print ("Accuracy on the training set: ", accuracy.eval(train_data))
	print ("Accuracy on the validation set: ", accuracy.eval(valid_data))
	print ("Accuracy on the testing set: ", accuracy.eval(test_data))


	test_results = sess.run(pred, feed_dict = {x:test_x})
	df_final = pd.DataFrame({'test_x':test_x[:,0],'pred': test_results[:,0]})

	df_loss = pd.DataFrame(t, columns = ['step', 'train_loss', 'valid_loss'])


'''
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15,5))
df.plot.scatter(x = 'x', y = 'y', ax = axes, color = 'red')
df_final.plot.scatter(x = 'test_x', y = 'pred', ax = axes, alpha = 0.3)

axes.set_title('Target vs Pred', fontsize = 20)
axes.set_xlabel('X', fontsize = 15)
axes.set_ylabel('y', fontsize = 15)
axes.legend(["target", "pred"], loc = 'best')
'''

df_loss.set_index('step').plot(logy = True, figsize = (15,5))	
plt.show()









































