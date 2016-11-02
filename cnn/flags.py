data_dir = 'data/cifar-10-batches-bin'
batch_size = 128

IMAGE_SIZE = 24
NUM_CLASSES = 10

#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
min_fraction_of_examples_in_queue = 0.4


NUM_EPOCHS_PER_DECAY = 350
INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.9999

train_dir = 'train'
#max_steps = 1000000 
max_steps = 10000
