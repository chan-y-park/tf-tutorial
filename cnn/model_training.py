import os
import time
from datetime import datetime
import tensorflow as tf
import numpy as np

import model_inputs
#from cifar10 import distorted_inputs 
from model_prediction import inference
#from cifar10 import inference

import flags


def get_total_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example',
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    losses = tf.get_collection('losses')
    loss_value = tf.add_n(losses, name='total_loss')
    loss_value = tf.Print(loss_value, [loss_value])
    return loss_value


def get_train_op(total_loss, global_step):
    num_batches_per_epoch = (
        flags.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / flags.batch_size
    )

    decay_steps = int(num_batches_per_epoch * flags.NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(
        flags.INITIAL_LEARNING_RATE, global_step, decay_steps,
        flags.LEARNING_RATE_DECAY_FACTOR, staircase=True,
    )
    tf.scalar_summary('learning_rate', lr)

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
        flags.MOVING_AVERAGE_DECAY, global_step,
    )
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def train():
    g = tf.Graph()
    with g.as_default():
        global_step = tf.Variable(0, trainable=False)

        images, labels = model_inputs.inputs()
        #images, labels = distorted_inputs()

        logits = inference(images)

        loss = get_total_loss(logits, labels)

        train_op = get_train_op(loss, global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(flags.train_dir, sess.graph)

        for step in range(flags.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            losses = tf.get_collection('losses')

            assert (not np.isnan(loss_value))

            if step % 10 == 0:
                num_examples_per_step = flags.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                print(
                    '{}: step {}, loss = {:.2f} '
                    '({:.1f} examples/sec; {:.3f} sec/batch)'
                    .format(datetime.now(), step, loss_value,
                            examples_per_sec, sec_per_batch)
                )

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step + 1) == flags.max_steps:
                checkpoint_path = os.path.join(flags.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

