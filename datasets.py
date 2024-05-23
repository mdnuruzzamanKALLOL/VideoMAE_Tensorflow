import os
import tensorflow as tf
import numpy as np

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.train_augmentation = self.group_multi_scale_crop(args.input_size, [1, .875, .75, .66])
        self.normalize = self.group_normalize(self.input_mean, self.input_std)
        self.masked_position_generator = None
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(args.window_size, args.mask_ratio)

    def group_normalize(self, mean, std):
        def normalize(images):
            mean_tensor = tf.constant(mean, dtype=tf.float32, shape=[1, 1, 1, 3])
            std_tensor = tf.constant(std, dtype=tf.float32, shape=[1, 1, 1, 3])
            return (images - mean_tensor) / std_tensor
        return normalize

    def group_multi_scale_crop(self, input_size, scales):
        def multi_scale_crop(images):
            scale = np.random.choice(scales)
            new_size = tf.cast(input_size * scale, tf.int32)
            cropped_images = tf.image.resize(images, (new_size, new_size))
            return tf.image.random_crop(cropped_images, [input_size, input_size, 3])
        return multi_scale_crop

    def __call__(self, images):
        process_data = self.train_augmentation(images)
        process_data = self.normalize(process_data)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.train_augmentation)
        repr += "  normalize = %s,\n" % str(self.normalize)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset

def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = 'train' if is_train else ('test' if test_mode else 'validation')
        anno_path = os.path.join(args.data_path, f'{mode}.csv')
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400

    elif args.data_set == 'SSV2':
        mode = 'train' if is_train else ('test' if test_mode else 'validation')
        anno_path = os.path.join(args.data_path, f'{mode}.csv')
        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = 'train' if is_train else ('test' if test_mode else 'validation')
        anno_path = os.path.join(args.data_path, f'{mode}.csv')
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101

    elif args.data_set == 'HMDB51':
        mode = 'train' if is_train else ('test' if test_mode else 'validation')
        anno_path = os.path.join(args.data_path, f'{mode}.csv')
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51

    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    return dataset, nb_classes
