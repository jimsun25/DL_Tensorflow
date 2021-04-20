import numpy as np
import tensorflow as tf

BATCH_SIZE = 16
ITER = 1000

def read_my_file_format(filename_queue):
  reader = tf.TextLineReader(skip_header_lines=1)
  key, value = reader.read(filename_queue)
  record_defaults = [[0.0], [0.0], [0.0], [0.0], ['']]
  
  col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
  
  features = tf.stack([col1, col2, col3, col4])
  return features, col5

def input_pipeline(filenames, batch_size):
  filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
      
  example, label = read_my_file_format(filename_queue)
  
  min_after_dequeue = 1000
  capacity = min_after_dequeue + 3 * batch_size
  
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
      
  return example_batch, label_batch

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

filenames = ["iris1.csv", "iris2.csv", "iris3.csv"]
example_bat, label_bat = input_pipeline(filenames, batch_size=BATCH_SIZE)

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 4])
ys = tf.placeholder(tf.float32, [None, 3])

# add output layer
L1 = add_layer(xs, 4, 100,  activation_function=tf.nn.tanh)
L2 = add_layer(L1, 100, 100,  activation_function=tf.nn.tanh)
prediction = add_layer(L2, 100, 3,  activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=[1])) # loss
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

saver = tf.train.Saver()

with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())
  
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
    
  for i in range(ITER):
    example, label = sess.run([example_bat, label_bat])
    #print(example, label)
    
    labelonehot = []
    for i in range (BATCH_SIZE):
      if label[i] == b'versicolor':
        labelonehot.append([1, 0, 0])
      elif label[i] == b'setosa':
        labelonehot.append([0, 1, 0])
      elif label[i] == b'virginica':
        labelonehot.append([0, 0, 1])
      else:
        print('Wrong Label:', label[i])
        exit(1)
    #print('One-Hot Label:', labelonehot)
    
    _, loss = sess.run([train_step, cross_entropy], feed_dict={xs: example, ys: labelonehot})
  
        
  print('Loss: ', loss)
  save_path = saver.save(sess, 'model/model.ckpt')
  print("Save to path: ", save_path)
  
  coord.request_stop()
  coord.join(threads)
  