import tensorflow as tf 

with tf.Session() as sess:
    filename = ['A.jpg', 'B.jpg', 'C.jpg']
    filename_queue = tf.train.string_input_producer(filename, shuffle=True, num_epochs=5)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess)

    i = 0
    while True:
        i += 1
        try:
            image_data = sess.run(value)
            with open('read/test_%d.jpg' % i, 'wb') as f:
                f.write(image_data)

        except tf.errors.OutOfRangeError:
            break

    print("Run Finished!")