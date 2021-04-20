import numpy as np
import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 4])
ys = tf.placeholder(tf.float32, [None, 3])

# add output layer
L1 = add_layer(xs, 4, 100,  activation_function=tf.nn.tanh)
L2 = add_layer(L1, 100, 100,  activation_function=tf.nn.tanh)
prediction = add_layer(L2, 100, 3,  activation_function=tf.nn.softmax)

saver = tf.train.Saver()

with tf.Session() as sess:
  print("Model loading...")
  saver.restore(sess, 'model/model.ckpt')
   
  ex = [[5.7,4.4,1.5,0.4], [6.9,3.1,4.9,1.5], [7.7,3,6.1,2.3], [6.7,3.1,4.4,1.4],
        [6,2.7,5.1,1.6],   [6.4,3.2,4.5,1.5], [6,3,4.8,1.8],   [7.2,3.2,6,1.8],
        [5.4,3,4.5,1.5],   [4.8,3.1,1.6,0.2], [6,2.2,5,1.5],   [5.4,3.4,1.5,0.4],
        [5,3,1.6,0.2],     [6.4,2.9,4.3,1.3], [6.7,3.3,5.7,2.5]] 
  
  lb = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0],
        [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0],
        [0, 1, 0], [1, 0, 0], [0, 0, 1]]
   
  #"setosa", "versicolor", "virginica", "versicolor",
  #"versicolor", "versicolor", "virginica", "virginica"
  #"versicolor", "setosa", "virginica", "setosa",
  #"setosa", "versicolor", "virginica"

  guess = sess.run(prediction, feed_dict={xs: ex})
  correct_prediction = tf.equal(tf.argmax(guess,1), tf.argmax(lb,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  result = sess.run(accuracy, feed_dict={xs: ex, ys: lb})
   
  print('accuracy: ', result)
