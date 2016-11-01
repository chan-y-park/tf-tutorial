import os
import math
import time
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def run(**kwargs):
    with tf.Graph().as_default():
        training(**kwargs)

def training(
    mnist_image_pixels=IMAGE_PIXELS,
    batch_size=100,
    num_hidden1_units=128,
    num_hidden2_units=32,
    num_classes=10,
    learning_rate=.01,
    train_dir='data',
    max_steps=2000,
):
    data_sets = read_data_sets(train_dir)

    images_placeholder = tf.placeholder(
        tf.float32,
        shape=(batch_size, mnist_image_pixels)
    )
    labels_placeholder = tf.placeholder(
        #tf.int32,
        tf.int64,
        shape=(batch_size)
    )

    with tf.name_scope('hidden1'):
        W, b = get_W_b(mnist_image_pixels, num_hidden1_units)
        hidden1 = tf.nn.relu(tf.matmul(images_placeholder, W) + b)

    with tf.name_scope('hidden2'):
        W, b = get_W_b(num_hidden1_units, num_hidden2_units)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, W) + b)

    with tf.name_scope('softmax_linear'):
        W, b = get_W_b(num_hidden2_units, num_classes)
        logits = tf.matmul(hidden2, W) + b

    #labels_placeholder = tf.to_int64(labels_placeholder)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels_placeholder, name='xentropy'
    )

    loss = tf.reduce_mean(cross_entropy, name='xentorpy_mean')

    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
    eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32)) 

    summary = tf.merge_all_summaries()

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    sess = tf.Session()

    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

    sess.run(init)

    for step in range(max_steps):
        start_time = time.time()

        feed_dict=get_feed_dict(
            data_sets.train, images_placeholder, labels_placeholder,
            batch_size
        )

        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

        duration = time.time() - start_time

        if step % 100 == 0:
            print('Step {:d}: loss = {:.2f} ({:.3f} sec)'
                  .format(step, loss_value, duration))
            # Update the events file.
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
            
        if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
            checkpoint_file = os.path.join(train_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=step)
            print('Training Data Eval:')
            do_eval(
                sess, eval_correct, images_placeholder,
                labels_placeholder, data_sets.train, batch_size
            )
            print('Validation Data Eval:')
            do_eval(
                sess, eval_correct, images_placeholder,
                labels_placeholder, data_sets.validation, batch_size
            )
            print('Test Data Eval:')
            do_eval(
                sess, eval_correct, images_placeholder,
                labels_placeholder, data_sets.test, batch_size
            )


def get_W_b(num_ins, num_outs):
    weights = tf.Variable(
        tf.truncated_normal(
            [num_ins, num_outs],
            stddev=(1.0 / math.sqrt(float(num_outs)))
        ),
        name='weights',
    )

    biases = tf.Variable(
        tf.zeros([num_outs]),
        name='biases',
    )

    return (weights, biases)


def get_feed_dict(data_set, images_pl, labels_pl, batch_size):
    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(
    sess, eval_correct, images_pl, labels_pl, data_set, batch_size    
):
    true_count = 0
    steps_per_epoch = data_set.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    for step in range(steps_per_epoch):
        feed_dict = get_feed_dict(
            data_set, images_pl, labels_pl, batch_size
        )

        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: {:d}  Num correct: {:d}  Precision @ 1: {:0.04f}'
          .format(num_examples, true_count, precision))
