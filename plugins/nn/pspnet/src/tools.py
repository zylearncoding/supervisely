import numpy as np
import tensorflow as tf


IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)


def decode_to_n_channels(mask, img_shape, num_classes):
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (1, img_shape[0], img_shape[1], num_classes))
    return onehot_output


def preprocess(img, h, w):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    pad_img = tf.image.pad_to_bounding_box(img, 0, 0, h, w)
    pad_img = tf.expand_dims(pad_img, dim=0)

    return pad_img
