import tensorflow as tf

def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[""], [""], [0], [0], [0], [0]]
    country, code, gold, silver, bronze, total = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.stack([gold, silver, bronze])
    return features, country

filenames = ["olympics2016.csv"]

filename_queue = tf.train.string_input_producer(filenames, num_epochs=2, shuffle=False)
example, country = create_file_reader_ops(filename_queue)

with tf.train.MonitoredSession() as sess:
    
     tf.train.start_queue_runners(sess=sess)
     
     while not sess.should_stop():
           example_data, country_name = sess.run([example, country])
           print(example_data, country_name)

     
     
     
