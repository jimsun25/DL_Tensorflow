import tensorflow as tf

c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d, name='example')

with tf.Session() as sess:
  test = sess.run(e)
  print(e.name)
  
  # example:0
  # <name>:0 (0 refers to endpoint which is somewhat redundant)
  test = tf.get_default_graph().get_tensor_by_name("example:0")
  
  print(test)
  # Tensor("example:0", shape=(2, 2), dtype=float32)
  
