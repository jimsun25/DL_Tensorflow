import tensorflow as tf # tensorflow module
import numpy as np      # numpy module
import os               # path join

import matplotlib.pyplot as plt

DATA_DIR = "./"
TRAINING_SET_SIZE = 3570
RUN_EPOCH = 100
BATCH_SIZE = 128
IMAGE_SIZE = 224
DROPOUT = 0.8
CLASS = 5

tf.app.flags.DEFINE_boolean('eval', False, '--eval=True for evaluation')
tf.app.flags.DEFINE_string('image', './data_test/test.jpg', '--image=path')

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# image object from protobuf
class _image_object:
    def __init__(self):
        self.image = tf.Variable([], dtype = tf.string)
        self.height = tf.Variable([], dtype = tf.int64)
        self.width = tf.Variable([], dtype = tf.int64)
        self.filename = tf.Variable([], dtype = tf.string)
        self.label = tf.Variable([], dtype = tf.int32)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})
    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)
    image_object = _image_object()
    image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    image_object.height = features["image/height"]
    image_object.width = features["image/width"]
    image_object.filename = features["image/filename"]
    image_object.label = tf.cast(features["image/class/label"], tf.int64)
    return image_object

def flower_input(if_random = True, if_training = True):
    if(if_training):
        filenames = [os.path.join(DATA_DIR, "train-0000%d-of-00002.tfrecord" % i) for i in range(0, 2)]
    else:
        filenames = [os.path.join(DATA_DIR, "validation-0000%d-of-00002.tfrecord" % i) for i in range(0, 2)]
        #filenames = [os.path.join(DATA_DIR, "train-0000%d-of-00002.tfrecord" % i) for i in range(0, 2)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)
    filename_queue = tf.train.string_input_producer(filenames)
    image_object = read_and_decode(filename_queue)
    image = tf.image.per_image_standardization(image_object.image)
    #image = image_object.image
    #image = tf.image.adjust_gamma(tf.cast(image_object.image, tf.float32), gamma=1, gain=1) # Scale image to (0, 1)
    label = image_object.label
    filename = image_object.filename

    if(if_random):
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
        print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
        num_preprocess_threads = 4
        image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue = min_queue_examples)
        return image_batch, label_batch, filename_batch
    else:
        image_batch, label_batch, filename_batch = tf.train.batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = 1)
        return image_batch, label_batch, filename_batch


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.02, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def flower_inference(image_batch, keep_prob):
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(image_batch, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) # IMAGE_SIZE/2

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # IMAGE_SIZE/4

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3) # IMAGE_SIZE/8

    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4) # IMAGE_SIZE/16

    W_conv5 = weight_variable([5, 5, 256, 256])
    b_conv5 = bias_variable([256])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5) # IMAGE_SIZE/32

    hp5shape = h_pool5.get_shape().as_list()
    h_pool5_flattened_size = hp5shape[1]*hp5shape[2]*hp5shape[3]

    W_fc1 = weight_variable([h_pool5_flattened_size, 2048])
    b_fc1 = bias_variable([2048])

    h_pool5_flat = tf.reshape(h_pool5, [-1, h_pool5_flattened_size])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([2048, 256])
    b_fc2 = bias_variable([256])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    W_fc3 = weight_variable([256, 64])
    b_fc3 = bias_variable([64])

    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    W_fc4 = weight_variable([64, CLASS])
    b_fc4 = bias_variable([CLASS])

    y_conv = tf.matmul(h_fc3, W_fc4) + b_fc4

    y_conv_softmax = tf.nn.softmax(y_conv)

    return y_conv, y_conv_softmax


def flower_train():
    image_batch_out, label_batch_out, filename_batch = flower_input(if_random = True, if_training = True)
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    label_batch_one_hot = tf.one_hot(tf.add(label_batch_out, label_offset), depth=CLASS, on_value=1.0, off_value=0.0)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    label_batch_placeholder = tf.placeholder(tf.float32, shape=[None, CLASS])
    keep_prob = tf.placeholder(tf.float32)
    
    logits_out, logits_out_softmax = flower_inference(image_batch_placeholder, keep_prob)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_placeholder, logits=logits_out))
#    loss = tf.reduce_mean(-tf.reduce_sum(label_batch_placeholder * tf.log(logits_out_softmax), reduction_indices=[1]))
#    loss = tf.losses.mean_squared_error(labels=label_batch_placeholder, predictions=logits_out_softmax)

    #train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        # Visualize the graph through tensorboard.
        file_writer = tf.summary.FileWriter("./logs", sess.graph)

        ckpt = tf.train.get_checkpoint_state('./checkpoint')
        if ckpt and ckpt.all_model_checkpoint_paths:
          model_path = tf.train.latest_checkpoint('./checkpoint')
          print("%s Model restoring..." % model_path)
          saver.restore(sess, model_path)
          #saver.restore(sess, "./checkpoint/checkpoint-train.ckpt")
        else:
          print("No model found, start a fresh new training")
          sess.run(tf.global_variables_initializer())
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        total_iteration = (TRAINING_SET_SIZE//BATCH_SIZE) * RUN_EPOCH

        for i in range(total_iteration):

            image_out, label_out, label_batch_one_hot_out, filename_out = sess.run([image_batch, label_batch_out, label_batch_one_hot, filename_batch])

            _, infer_out, loss_out = sess.run([train_step, logits_out, loss], feed_dict={image_batch_placeholder: image_out, 
                                                                                         label_batch_placeholder: label_batch_one_hot_out, 
                                                                                         keep_prob: DROPOUT})

            print("%d/%d: loss: %f" % (i, total_iteration, loss_out))
            
            #print(i)
            #print(image_out.shape)
            #print("label_out: ")
            #print(filename_out)
            #print(label_out)
            #print(label_batch_one_hot_out)
            #print("infer_out: ")
            #print(infer_out)
            
            if(i%500 == 0):
                saver.save(sess, "./checkpoint/train", global_step=i)
                #saver.save(sess, "./checkpoint/checkpoint-train.ckpt", global_step=i)
                print("model saved at iteration: %d" % i)

        saver.save(sess, "./checkpoint/train", global_step=i)
        print("last model saved at iteration: %d" % i)

        coord.request_stop()
        coord.join(threads)


def flower_eval():
    image_batch_out, label_batch_out, filename_batch = flower_input(if_random = False, if_training = False)
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    label_batch_one_hot = tf.one_hot(tf.add(label_batch_out, label_offset), depth=CLASS, on_value=1.0, off_value=0.0)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    label_batch_placeholder = tf.placeholder(tf.float32, shape=[None, CLASS])
    keep_prob = tf.placeholder(tf.float32)
    
    _, logits_out_softmax = flower_inference(image_batch_placeholder, keep_prob)

    logits_max = tf.argmax(logits_out_softmax, axis = 1)
    labels_max = tf.argmax(label_batch_placeholder, axis = 1)

    correct_prediction = tf.equal(logits_max, labels_max)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
    
        ckpt = tf.train.get_checkpoint_state('./checkpoint')
        if ckpt and ckpt.all_model_checkpoint_paths:
          model_path = tf.train.latest_checkpoint('./checkpoint')
          print("%s Model restoring..." % model_path)
          saver.restore(sess, model_path)
          #saver.restore(sess, "./checkpoint/checkpoint-train.ckpt")
        else:
          print("Error: No model found for evaluation")
          exit(1)
        
        image_data = tf.gfile.FastGFile(FLAGS.image, 'rb').read()
        decoded_image_data = tf.image.decode_jpeg(image_data, channels=3)

        #resized_image = tf.image.resize_images(decoded_image_data, [IMAGE_SIZE, IMAGE_SIZE])
        resized_image = tf.image.resize_image_with_crop_or_pad(decoded_image_data, IMAGE_SIZE, IMAGE_SIZE)

        input_image_std = tf.image.per_image_standardization(resized_image)
        input_image_reshaped = tf.reshape(input_image_std, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

        input_image = sess.run(input_image_reshaped)

        image_index = sess.run(logits_max, feed_dict={image_batch_placeholder:input_image, keep_prob: 1})

        if image_index == 0:
          class_label = 'daisy'
        elif image_index == 1:
          class_label = 'tulips'
        elif image_index == 2:
          class_label = 'dandelion'
        elif image_index == 3:
          class_label = 'roses'
        elif image_index == 4:
          class_label = 'sunflowers'
        else:
          class_label = 'Wrong Prediction: ' + str(image_index)
        
        print("image %s is --> %s" % (FLAGS.image, class_label))

        input_image = sess.run(decoded_image_data)
        plt.imshow(input_image)
        plt.title = (class_label)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
  flower_eval()

