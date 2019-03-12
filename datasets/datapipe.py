import os, math, random
from glob import glob
from libs.configs import config
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def _flip_image(image):
    return tf.reverse(image, axis=[1])

def _rotate_images(image, gt_mask, seed):
    ratio = seed * 2.0 - 1.0
    angle = (ratio*30.0) * (math.pi / 180.0)
    rotate_image = tf.contrib.image.rotate(image, angle)
    rotate_mask = tf.contrib.image.rotate(gt_mask, angle)
    return rotate_image, rotate_mask


def _preprocess_for_training(dataA, dataB):
    resize_A = tf.image.resize_images(dataA, [FLAGS.image_size, FLAGS.image_size])
    resize_B = tf.image.resize_images(dataB, [FLAGS.image_size, FLAGS.image_size])

    norm_A = tf.cast(resize_A, tf.float32) /127.5 - 1.0
    norm_B = tf.cast(resize_B, tf.float32) /127.5 - 1.0

    seed = random.randint(0, 2**31-1)
    norm_A = tf.image.random_flip_left_right(norm_A, seed=seed)
    norm_B = tf.image.random_flip_left_right(norm_B, seed=seed)

    return norm_A, norm_B

def _preprocess_for_test(image):

    resize_img = tf.image.resize_images(image, [FLAGS.image_size, FLAGS.image_size])

    norm_img = tf.cast(resize_img, tf.float32) /127.5 - 1.0

    norm_img = tf.expand_dims(norm_img, 0)
    print(norm_img)
    return norm_img

def get_dataset():
    base_dir = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name)

    print(base_dir)
    trainA_dataset = sorted(glob(os.path.join(base_dir, 'trainA/*')))
    trainB_dataset = sorted(glob(os.path.join(base_dir, 'trainB/*')))
    print(len(trainA_dataset), len(trainB_dataset), trainA_dataset[0], trainB_dataset[0])
    num_dataset = max(len(trainA_dataset), len(trainB_dataset))
    print(num_dataset)

    domainA = tf.convert_to_tensor(trainA_dataset)
    domainB = tf.convert_to_tensor(trainB_dataset)
    inputA_queue = tf.train.slice_input_producer([domainA], shuffle=True, name='inputA_queue')
    inputB_queue = tf.train.slice_input_producer([domainB], shuffle=True, name='inputB_queue')
    dataA_fn = tf.read_file(inputA_queue[0], name='read_imageA')
    dataB_fn = tf.read_file(inputB_queue[0], name='read_imageB')

    dataA = tf.image.decode_jpeg(dataA_fn, channels=3, name='decode_imageA')
    dataB = tf.image.decode_jpeg(dataB_fn, channels=3, name='decode_imageB')

    preprocessed_domainA, preprocessed_domainB = _preprocess_for_training(dataA, dataB)
    batchs_A = tf.train.shuffle_batch([preprocessed_domainA],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity=512,
                                      min_after_dequeue=300)

    batchs_B = tf.train.shuffle_batch([preprocessed_domainB],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity=512,
                                      min_after_dequeue=300)

    batch_domainA, batch_domainB = batchs_A, batchs_B

    return batch_domainA, batch_domainB


