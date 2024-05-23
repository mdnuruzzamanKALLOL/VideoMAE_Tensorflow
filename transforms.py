import tensorflow as tf
import numpy as np

class GroupRandomCrop:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img_group, label):
        h, w = tf.shape(img_group[0])[0], tf.shape(img_group[0])[1]
        th, tw = self.size

        x1 = tf.random.uniform([], 0, w - tw + 1, dtype=tf.int32)
        y1 = tf.random.uniform([], 0, h - th + 1, dtype=tf.int32)

        out_images = [tf.image.crop_to_bounding_box(img, y1, x1, th, tw) for img in img_group]

        return out_images, label

class GroupCenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img_group, label):
        return [tf.image.central_crop(img, self.size[0] / tf.shape(img)[0]) for img in img_group], label

class GroupNormalize:
    def __init__(self, mean, std):
        self.mean = tf.constant(mean, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)

    def __call__(self, tensor, label):
        return [(img - self.mean) / self.std for img in tensor], label

class GroupGrayScale:
    def __init__(self):
        pass

    def __call__(self, img_group, label):
        return [tf.image.rgb_to_grayscale(img) for img in img_group], label

class GroupScale:
    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group, label):
        return [tf.image.resize(img, self.size, method=self.interpolation) for img in img_group], label

class GroupMultiScaleCrop:
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]

    def __call__(self, img_group, label):
        im_size = tf.shape(img_group[0])

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [tf.image.crop_to_bounding_box(img, offset_h, offset_w, crop_h, crop_w) for img in img_group]
        ret_img_group = [tf.image.resize(img, self.input_size) for img in crop_img_group]
        return ret_img_group, label

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[1], im_size[0]

        base_size = tf.minimum(image_w, image_h)
        crop_sizes = [tf.cast(base_size * x, tf.int32) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = tf.random.uniform([], 0, image_w - crop_pair[0] + 1, dtype=tf.int32)
            h_offset = tf.random.uniform([], 0, image_h - crop_pair[1] + 1, dtype=tf.int32)
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = [
            (0, 0),  # upper left
            (4 * w_step, 0),  # upper right
            (0, 4 * h_step),  # lower left
            (4 * w_step, 4 * h_step),  # lower right
            (2 * w_step, 2 * h_step)  # center
        ]

        if more_fix_crop:
            ret.extend([
                (0, 2 * h_step),  # center left
                (4 * w_step, 2 * h_step),  # center right
                (2 * w_step, 4 * h_step),  # lower center
                (2 * w_step, 0 * h_step),  # upper center
                (1 * w_step, 1 * h_step),  # upper left quarter
                (3 * w_step, 1 * h_step),  # upper right quarter
                (1 * w_step, 3 * h_step),  # lower left quarter
                (3 * w_step, 3 * h_step)  # lower right quarter
            ])
        return ret

class Stack:
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group, label):
        if img_group[0].shape[-1] == 1:  # Grayscale image
            return np.concatenate([np.expand_dims(img, axis=-1) for img in img_group], axis=-1), label
        elif img_group[0].shape[-1] == 3:  # RGB image
            if self.roll:
                return np.concatenate([img[:, :, ::-1] for img in img_group], axis=-1), label
            else:
                return np.concatenate(img_group, axis=-1), label

class ToTensor:
    def __init__(self, div=True):
        self.div = div

    def __call__(self, img_group, label):
        img_tensor = [tf.convert_to_tensor(img, dtype=tf.float32) for img in img_group]
        img_tensor = [tf.image.per_image_standardization(img) for img in img_tensor]
        if self.div:
            img_tensor = [img / 255.0 for img in img_tensor]
        return img_tensor, label

class IdentityTransform:
    def __call__(self, data):
        return data

# Example usage
transforms = tf.keras.Sequential([
    GroupRandomCrop(size=(224, 224)),
    GroupCenterCrop(size=224),
    GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    GroupGrayScale(),
    GroupScale(size=(256, 256)),
    GroupMultiScaleCrop(input_size=224),
    Stack(roll=False),
    ToTensor(div=True)
])

# Example video tensor with shape (frames, height, width, channels)
video = tf.random.uniform((8, 128, 128, 3))
label = 1  # Example label
transformed_video, transformed_label = transforms((video, label))