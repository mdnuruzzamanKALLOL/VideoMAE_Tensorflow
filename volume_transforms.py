import numpy as np
import tensorflow as tf
from PIL import Image

def convert_img(img):
    if len(img.shape) == 3:
        img = np.transpose(img, (2, 0, 1))
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img

class ClipToTensor(object):
    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, 'Got {0} instead of 3 channels'.format(ch)
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))

        np_clip = np.zeros([self.channel_nb, len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))
            img = convert_img(img)
            np_clip[:, img_idx, :, :] = img
        if self.numpy:
            if self.div_255:
                np_clip = np_clip / 255.0
            return np_clip
        else:
            tensor_clip = tf.convert_to_tensor(np_clip, dtype=tf.float32)
            if self.div_255:
                tensor_clip = tensor_clip / 255.0
            return tensor_clip

class ToTensor(object):
    """Converts numpy array to tensor"""

    def __call__(self, array):
        tensor = tf.convert_to_tensor(array, dtype=tf.float32)
        return tensor
