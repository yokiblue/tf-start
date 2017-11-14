import tensorflow as tf 
import pandas as pd 
import numpy as np 
import sys
import datetime
import matplotlib.pyplot as plt 
plt.style.use('ggplot')


# function y = a * x^4 + b
pool = np.random.rand(1000, 1).astype(np.float32)
np.random.shuffle(pool)
sample = int(1000 * 0.15)
test_x = pool[0 : sample]
valid_x = pool[sample : sample * 2]
train_x = pool[sample*2 :]

print ("Testing / validation / training sets are: ", 
	str(test_x.shape), str(valid_x.shape), str(train_x.shape))

test_y = 2.0 * test_x ** 4 + 5
valid_y = 2.0 * valid_x ** 4 + 5
train_y = 2.0 * train_x ** 4 + 5

df = pd.DataFrame({'x': train_x[:,0], 'y': train_y[:,0]})

#print (df.head())
#print (df.describe())
#df.plot.scatter(x = 'x', y = 'y', figsize = (15, 5))
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

batch_size = 10

hidden_size = 10
x = tf.placeholder(tf.float32, shape = [None, 1], name = "01_x")
y = tf.placeholder(tf.float32, shape = [None, 1], name = "01_y")

print ("Shape of x and y: ", x.get_shape(), y.get_shape())

# drop out
keep_prob = tf.placeholder(tf.float32)

# create hidden layers
h1 = add_layer(x, 1, hidden_size, tf.nn.relu)
h1_drop = tf.nn.dropout(h1, keep_prob)

h2 = add_layer(h1_drop, hidden_size, hidden_size, tf.nn.relu)
h2_drop = tf.nn.dropout(h2, keep_prob)

h3 = add_layer(h2_drop, hidden_size, hidden_size, tf.nn.relu)
h3_drop = tf.nn.dropout(h3, keep_prob)

h4 = add_layer(h3_drop, hidden_size, hidden_size, tf.nn.relu)
h4_drop = tf.nn.dropout(h4, keep_prob)


print ("Shape of hidden layers: ")
print (h1_drop.get_shape(), h2_drop.get_shape(), h3_drop.get_shape(), h4_drop.get_shape())

# create output layer

pred = add_layer(h4_drop, hidden_size, 1)

print ("Shape of output layer: ", pred.get_shape())

loss = tf.reduce_mean(tf.square(pred - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.round(pred), tf.round(y))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

best_valid_acc = 0.0
last_improvement = 0
require_improvement = 1500


# initialize the variables
init = tf.global_variables_initializer()

# hold step and error values
t = []

# Run your graph
with tf.Session() as sess:
    
    # initialize variables
    sess.run(init)

    # Fit the function.
    for step in range(6000):
        
        # pull batches at random
        i = np.random.permutation(train_x.shape[0])[:batch_size]

        # get your data
        train_data = {x:train_x[i,:], y:train_y[i,:], keep_prob: 0.975}
        valid_data = {x:valid_x, y:valid_y, keep_prob: 1.0}
        test_data = {x:test_x, y:test_y, keep_prob: 1.0}
        
        # training in progress...
        train_loss, train_pred = sess.run([loss, train], feed_dict=train_data)        
        
        # print every n iterations
        if step%100==0:
           
            # capture the step and error for analysis
            valid_loss = sess.run(loss, feed_dict=valid_data) 
            t.append((step, train_loss, valid_loss))    
            
            # get snapshot of current training and validation accuracy       
            train_acc = accuracy.eval(train_data)
            valid_acc = accuracy.eval(valid_data)           

            # If validation accuracy is an improvement over best-known.
            if valid_acc > best_valid_acc:
                # Update the best-known validation accuracy.
                best_valid_acc = valid_acc
                
                # Set the iteration for the last improvement to current.
                last_improvement = step

                # Flag when ever an improvement is found
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''   
                
            print("Training loss at step %d: %f %s" % (step, train_loss, improved_str))        
            print("Validation %f" % (valid_loss))            
                
            # If no improvement found in the required number of iterations.
            if step - last_improvement > require_improvement:
                print("No improvement found in a while, stopping optimization.")

                # Break out from the for-loop.
                break                
            
            
    # here is where you see how good of a Data Scientist you are        
    print("Accuracy on the Training Set:", accuracy.eval(train_data) )
    print("Accuracy on the Validation Set:", accuracy.eval(valid_data) ) 
    print("Accuracy on the Test Set:", accuracy.eval(test_data) )
    
    # capture predictions on test data 
    test_results = sess.run(pred, feed_dict={x:test_x, keep_prob: 1.0})  
    df_final = pd.DataFrame({'test_x':test_x[:,0],
                             'pred':test_results[:,0]})
    
    # capture training and validation loss
    df_loss = pd.DataFrame(t, columns=['step', 'train_loss', 'valid_loss'])

df_loss.set_index('step').plot(logy=True, figsize=(15,5));   
plt.show()
















