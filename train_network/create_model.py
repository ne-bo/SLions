import os
import pathlib
import numpy as np
import tensorflow as tf
import datetime

batch_size = 7
num_labels = 6
num_epoch = 150

crop_height = 100
crop_width = 100
num_channels = 3


def read_my_file_format(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'image_and_labels_raw': tf.FixedLenFeature([], tf.string)},
        name='name_for_operation_parse_single_example')

    image_with_labels = tf.cast(tf.decode_raw(features['image_and_labels_raw'], tf.float64,
                                              name='name_for_decode_raw'), tf.float32)

    with tf.name_scope('image_size'):
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        depth = tf.cast(features['depth'], tf.int32)
        tf.summary.scalar('image_size_height', height)
        tf.summary.scalar('image_size_width', width)
        tf.summary.scalar('image_size_depth', depth)

    tf.Print(height, [height, width, depth], message='Finally print desired image size!!!')

    reshaped_image_with_labels = tf.reshape(image_with_labels,
                                            tf.convert_to_tensor(
                                                [height, width, depth]))

    cropped_example = tf.random_crop(reshaped_image_with_labels, [crop_height, crop_width, 4])
    processed_example = cropped_example[:, :, 0:3]

    label = tf.cast(cropped_example[int(crop_height / 2), int(crop_width / 2), 3], tf.int32)
    label_one_hot = tf.one_hot(label, on_value=1.0, off_value=0.0, depth=num_labels)
    tf.summary.scalar('label', label)
    tf.summary.scalar('label_one_hot', tf.reduce_mean(label_one_hot) * num_labels)
    return processed_example, label_one_hot


def input_pipeline(file_names, batch_size, num_epochs):
    filename_queue = tf.train.string_input_producer(file_names, num_epochs=num_epochs, shuffle=True)
    example, label = read_my_file_format(filename_queue)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 1
    capacity = min_after_dequeue + 3 * batch_size

    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        num_threads=2,
        min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)
    tf.summary.image('example_batch', example_batch)
    tf.summary.scalar('mean_label_of_the_batch', tf.reduce_mean(label_batch))
    return example_batch, label_batch


def get_new_data_batch():
#    if type_of_batch == 'train':
#        file_names = [("../get_data/train/natasha%d.jpg.npy.tfrecords" % i) for i in range(41, 48)]
#    if type_of_batch == 'validation':
#        file_names = [("../get_data/validation/natasha%d.jpg.npy.tfrecords" % i) for i in range(48, 50)]
#    if type_of_batch == 'test':
#        file_names = [("../get_data/test/natasha%d.jpg.npy.tfrecords" % i) for i in range(50, 51)]

#    file_names = [("../get_data/" + type + "/natasha%d.jpg.npy.tfrecords" % i) for i in range(41, 51)]
    #file_names = tf.train.match_filenames_once("../get_data/" + type + "/*.tfrecords")
    file_names = [("../get_data/train/natasha%d.jpg.npy.tfrecords" % i) for i in range(41, 48)]
    print("file_names = ", file_names)
    print("num_epoch = ", num_epoch)
    return input_pipeline(file_names, batch_size, num_epoch)

# something like main





example_batch_train, label_batch_train = get_new_data_batch()  # here we just describe our graph
#example_batch_validation, label_batch_validation = get_new_data_batch('validation')  # here we just describe our graph
#example_batch_test, label_batch_test = get_new_data_batch('test')  # here we just describe our graph


# Variables.
weights = tf.Variable(tf.truncated_normal([crop_height * crop_width * num_channels, num_labels]))
biases = tf.Variable(tf.zeros([num_labels]))

reshaped_example_batch_train = tf.reshape(example_batch_train,
                                          [batch_size, crop_width * crop_height * num_channels])
# Training computation.
logits = tf.matmul(reshaped_example_batch_train, weights) + biases
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_batch_train))
tf.summary.scalar('loss', loss)

# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(logits)
#valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
#test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


# with tf.name_scope('accuracy'):
#     train_prediction[:, tf.arg_max(train_prediction, 1)] = 1
#     tf.summary.scalar('Minibatch train accuracy',
#                       tf.contrib.metrics.accuracy(train_prediction,
#                                                   label_batch_train))
#     tf.summary.scalar('Validation accuracy',
#                       tf.contrib.metrics.accuracy(valid_prediction.eval(),
#                                                   label_batch_validation))
#     tf.summary.scalar('Test accuracy',
#                       tf.contrib.metrics.accuracy(test_prediction.eval(),
#                                                   label_batch_test))


print('Current directory ', os.getcwd())
print('Files in ../get_data', list(pathlib.Path('../get_data').glob('*')))
sess = tf.Session()

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

sess.run(init_op)

# Merge all the summaries and write them out to current dir
merged = tf.summary.merge_all()
writer_graph = tf.summary.FileWriter('.', sess.graph)


# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

i = 0
try:
    while not coord.should_stop():
        print(datetime.datetime.now())
        print('i = ', i)
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        summary, o, l, t = sess.run([merged, optimizer, loss, train_prediction])
        #summary, l, valid_prediction = sess.run([merged, valid_prediction])
        #summary, l, test_prediction = sess.run([merged, test_prediction])
        if i % 2 == 0:
            writer_graph.add_summary(summary, i)
        i = i + 1
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()
    writer_graph.close()
# Wait for threads to finish.
coord.join(threads)
sess.close()
