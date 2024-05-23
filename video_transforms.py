import tensorflow as tf
import random

class Resize:
    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, video):
        return tf.image.resize(video, self.size, method=self.interpolation)

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        return tf.image.central_crop(video, self.size[0] / tf.shape(video)[1])

class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        return tf.image.random_crop(video, size=(tf.shape(video)[0], self.size[0], self.size[1], tf.shape(video)[-1]))

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, video):
        def flip(video):
            return tf.image.flip_left_right(video)
        
        return tf.cond(tf.random.uniform([]) < self.prob, lambda: flip(video), lambda: video)

class Normalize:
    def __init__(self, mean, std):
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)

    def __call__(self, video):
        video = tf.cast(video, tf.float32) / 255.0
        return (video - self.mean) / self.std

class RandomErasing:
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, video):
        if tf.random.uniform([]) > self.probability:
            return video

        video_h, video_w, video_c = tf.shape(video)[1], tf.shape(video)[2], tf.shape(video)[3]
        video_area = video_h * video_w

        while True:
            erase_area = tf.random.uniform([], minval=self.sl, maxval=self.sh) * video_area
            aspect_ratio = tf.random.uniform([], minval=self.r1, maxval=1/self.r1)

            h = tf.cast(tf.round(tf.sqrt(erase_area * aspect_ratio)), tf.int32)
            w = tf.cast(tf.round(tf.sqrt(erase_area / aspect_ratio)), tf.int32)

            if h < video_h and w < video_w:
                x1 = tf.random.uniform([], minval=0, maxval=video_w - w, dtype=tf.int32)
                y1 = tf.random.uniform([], minval=0, maxval=video_h - h, dtype=tf.int32)
                video = self.erase(video, x1, y1, h, w)
                break

        return video

    def erase(self, video, x, y, h, w):
        erase_area = tf.zeros((tf.shape(video)[0], h, w, video.shape[3]))
        video = tf.tensor_scatter_nd_update(video, tf.stack([[i, y + j, x + k] for i in range(tf.shape(video)[0]) for j in range(h) for k in range(w)], axis=0), erase_area)
        return video

class ClipToTensor:
    def __call__(self, video):
        return tf.convert_to_tensor(video, dtype=tf.float32)

class RandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, video):
        def get_params(img):
            area = tf.shape(img)[0] * tf.shape(img)[1]
            for _ in range(10):
                target_area = area * tf.random.uniform([], minval=self.scale[0], maxval=self.scale[1])
                aspect_ratio = tf.random.uniform([], minval=self.ratio[0], maxval=self.ratio[1])

                w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
                h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)

                if tf.random.uniform([]) < 0.5:
                    w, h = h, w

                if h <= tf.shape(img)[0] and w <= tf.shape(img)[1]:
                    top = tf.random.uniform([], minval=0, maxval=tf.shape(img)[0] - h, dtype=tf.int32)
                    left = tf.random.uniform([], minval=0, maxval=tf.shape(img)[1] - w, dtype=tf.int32)
                    return top, left, h, w

            in_ratio = tf.shape(img)[1] / tf.shape(img)[0]
            if in_ratio < min(self.ratio):
                w = tf.shape(img)[1]
                h = tf.cast(tf.round(w / min(self.ratio)), tf.int32)
            elif in_ratio > max(self.ratio):
                h = tf.shape(img)[0]
                w = tf.cast(tf.round(h * max(self.ratio)), tf.int32)
            else:
                w = tf.shape(img)[1]
                h = tf.shape(img)[0]
            top = (tf.shape(img)[0] - h) // 2
            left = (tf.shape(img)[1] - w) // 2
            return top, left, h, w

        top, left, h, w = get_params(video[0])
        video = tf.image.crop_to_bounding_box(video, top, left, h, w)
        video = tf.image.resize(video, self.size)
        return video

# Example usage
video_transform = tf.keras.Sequential([
    Resize((256, 256)),
    CenterCrop((224, 224)),
    RandomHorizontalFlip(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ClipToTensor()
])

# Example video tensor with shape (frames, height, width, channels)
video = tf.random.uniform((8, 128, 128, 3))
transformed_video = video_transform(video)
