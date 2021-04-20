import tensorflow as tf

def read_my_file_format(filename_queue):
  reader = tf.TextLineReader(skip_header_lines=1)
  key, value = reader.read(filename_queue)
  record_defaults = [[0.0], [0.0], [0.0], [0.0], ['']]
  
  col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
    
  features = tf.stack([col1, col2, col3, col4])
  return features, col5

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
      
  example, label = read_my_file_format(filename_queue)
  
  min_after_dequeue = 1000
  capacity = min_after_dequeue + 3 * batch_size
  
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
      
  return example_batch, label_batch

filenames = ["iris1.csv", "iris2.csv", "iris3.csv"]
example_bat, label_bat = input_pipeline(filenames, batch_size=5, num_epochs=2)


with tf.train.MonitoredSession() as sess:
    
  tf.train.start_queue_runners(sess=sess)
  
  while not sess.should_stop():
    example, label = sess.run([example_bat, label_bat])
    print(example, label)


