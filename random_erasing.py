import tensorflow as tf
import numpy as np

class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if tf.random.uniform([]) > self.probability:
            return img

        img_h, img_w, img_c = img.shape
        img_area = img_h * img_w

        while True:
            erase_area = tf.random.uniform([], minval=self.sl, maxval=self.sh) * img_area
            aspect_ratio = tf.random.uniform([], minval=self.r1, maxval=1/self.r1)

            h = tf.cast(tf.round(tf.sqrt(erase_area * aspect_ratio)), tf.int32)
            w = tf.cast(tf.round(tf.sqrt(erase_area / aspect_ratio)), tf.int32)

            if h < img_h and w < img_w:
                x1 = tf.random.uniform([], minval=0, maxval=img_w - w, dtype=tf.int32)
                y1 = tf.random.uniform([], minval=0, maxval=img_h - h, dtype=tf.int32)
                img = self.erase(img, x1, y1, h, w)
                break

        return img

    def erase(self, img, x, y, h, w):
        mean = tf.reduce_mean(img, axis=[0, 1], keepdims=True)
        erase_area = tf.zeros((h, w, img.shape[2]))
        img = tf.tensor_scatter_nd_update(img, tf.stack([[y + i, x + j] for i in range(h) for j in range(w)]), erase_area)
        return img

def apply_random_erasing(images, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
    re = RandomErasing(probability, sl, sh, r1)
    return tf.map_fn(lambda img: re(img), images)
