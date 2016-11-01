import os
import tensorflow as tf
import flags


class CIFAR10Record(object):
    def __init__(self):
        self.height = 32
        self.width = 32
        self.depth = 3
        self.key = None
        self.label = None
        self.uint8image = None
    
    def get_image_bytes(self):
        return self.height * self.width * self.depth

def read_cifar10(filename_queue):
    result = CIFAR10Record()
    label_bytes = 1
    image_bytes = result.get_image_bytes()
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32
    )

    depth_major = tf.reshape(
        tf.slice(record_bytes, [label_bytes], [image_bytes]),
        [result.depth, result.height, result.width]
    )
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def inputs(
    data_dir=flags.data_dir,
    batch_size=flags.batch_size,
    distort=True,
):
    filenames = [os.path.join(data_dir, 'data_batch_{}.bin'.format(i))
                 for i in range(1, 6)]
    for f in filenames:
        if not os.path.exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    label = read_input.label
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = flags.IMAGE_SIZE
    width = flags.IMAGE_SIZE

    if distort:
        image = tf.random_crop(reshaped_image, [height, width, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    else:
        image = tf.image.resize_image_with_crop_or_pad(
            reshaped_image, width, height,
        )

    float_image = tf.image.per_image_whitening(image)

    min_queue_examples = int(flags.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             flags.min_fraction_of_examples_in_queue)
    print('Filling queue with {} CIFAR images before starting to train. '
          'This will take a few minutes.'.format(min_queue_examples))

    kwargs = {
        'batch_size': batch_size,
        'num_threads': 16,
        'capacity': min_queue_examples + 3 * batch_size,
    }
    if distort:
        kwargs['min_after_dequeue'] = min_queue_examples
        images, label_batch = tf.train.shuffle_batch([image, label], **kwargs)
    else:
        images, label_batch = tf.train.batch([image, label], **kwargs)

    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])
