import re
import tensorflow as tf
import flags

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=tf.float32
        )
    return var


def _variable_with_weight_decay(name, shape, stddev, wd=None):
    var = _variable_on_cpu(
        name, shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
    )
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _activation_summary(x):
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def inference(images):
    with tf.variable_scope('conv1') as scope:
        # XXX Why zero weight decay?
        kernel = _variable_with_weight_decay(
            'weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0,
        )
        conv = tf.nn.conv2d(
            images, kernel, [1, 1, 1, 1], padding='SAME',
        )
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    pool1 = tf.nn.max_pool(
        conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
        padding='SAME', name='pool1',
    )

    # XXX: Remove the following layer.
    # http://stackoverflow.com/questions/37376861/what-does-the-tf-nn-lrn-method-do
    norm1 = tf.nn.local_response_normalization(
        pool1, bias=1.0, alpha=(0.001 / 9.0), beta=0.75, name='norm1'
    )

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0,
        )
        conv = tf.nn.conv2d(
            norm1, kernel, [1, 1, 1, 1], padding='SAME',
        )
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    norm2 = tf.nn.local_response_normalization(
        conv2, depth_radius=4, bias=1.0, alpha=(0.001 / 9.0), beta=0.75,
        name='norm2',
    )

    pool2 = tf.nn.max_pool(
        norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
        name='pool2',
    )

    local3_num_nodes = 384
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [flags.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay(
            'weights', shape=[dim, local3_num_nodes], stddev=0.04, wd=0.004,
        )
        biases = _variable_on_cpu(
            'biases', [local3_num_nodes], tf.constant_initializer(0.1),
        )
        local3 = tf.nn.relu(
            tf.matmul(reshape, weights) + biases, name=scope.name,
        )
        _activation_summary(local3)


    local4_num_nodes = 192 
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay(
            'weights', shape=[local3_num_nodes, local4_num_nodes],
            stddev=0.04, wd=0.004,
        )
        biases = _variable_on_cpu(
            'biases', [local4_num_nodes], tf.constant_initializer(0.1),
        )
        local4 = tf.nn.relu(
            tf.matmul(local3, weights) + biases, name=scope.name,
        )
        _activation_summary(local4)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights', [local4_num_nodes, flags.NUM_CLASSES],
            stddev=(1.0 / local4_num_nodes), wd=0.0,
        )
        biases = _variable_on_cpu(
            'biases', [flags.NUM_CLASSES], tf.constant_initializer(0.0),
        )
        softmax_linear = tf.add(
            tf.matmul(local4, weights), biases, name=scope.name,
        )
        _activation_summary(softmax_linear)

    return softmax_linear
