#pip install tensorflow tensorflow-datasets

import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
CLIP_LEN = 8
FRAME_SAMPLE_RATE = 2

class VideoClsDataset(tf.keras.utils.Sequence):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop

        self.dataset_samples, self.label_array = self._load_annotations()
        self.data_transform = self._build_transforms()

    def _load_annotations(self):
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        dataset_samples = list(cleaned.values[:, 0])
        label_array = list(cleaned.values[:, 1])
        return dataset_samples, label_array

    def _build_transforms(self):
        if self.mode == 'train':
            return tf.keras.Sequential([
                layers.Resizing(self.short_side_size, self.short_side_size, interpolation='bilinear'),
                layers.RandomCrop(self.crop_size, self.crop_size),
                layers.RandomFlip('horizontal'),
                layers.Rescaling(1./255),
                layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
            ])
        elif self.mode == 'validation':
            return tf.keras.Sequential([
                layers.Resizing(self.short_side_size, self.short_side_size, interpolation='bilinear'),
                layers.CenterCrop(self.crop_size, self.crop_size),
                layers.Rescaling(1./255),
                layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
            ])
        elif self.mode == 'test':
            return tf.keras.Sequential([
                layers.Resizing(self.short_side_size, self.short_side_size, interpolation='bilinear'),
                layers.Rescaling(1./255),
                layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.dataset_samples) // BATCH_SIZE

    def __getitem__(self, index):
        batch_samples = self.dataset_samples[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
        batch_labels = self.label_array[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]

        batch_videos = [self.load_video(sample) for sample in batch_samples]
        batch_videos = tf.stack(batch_videos, axis=0)

        batch_videos = self.data_transform(batch_videos)
        batch_labels = tf.convert_to_tensor(batch_labels)

        return batch_videos, batch_labels

    def load_video(self, sample):
        video_path = os.path.join(self.data_path, sample)
        video_reader = tf.io.read_file(video_path)
        video = tf.io.decode_video(video_reader)

        # Sample frames from the video
        video = video[::self.frame_sample_rate]
        if len(video) < self.clip_len:
            pad_len = self.clip_len - len(video)
            paddings = tf.constant([[0, pad_len], [0, 0], [0, 0], [0, 0]])
            video = tf.pad(video, paddings)

        video = video[:self.clip_len]
        return video

def build_model(num_classes):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

def train_model(model, train_dataset, val_dataset):
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset
    )
    return history

def main():
    train_dataset = VideoClsDataset(anno_path='path/to/train_annotations.txt', data_path='path/to/train_videos', mode='train')
    val_dataset = VideoClsDataset(anno_path='path/to/val_annotations.txt', data_path='path/to/val_videos', mode='validation')

    train_dataset = tf.data.Dataset.from_generator(lambda: train_dataset, output_types=(tf.float32, tf.int32), output_shapes=((BATCH_SIZE, CLIP_LEN, IMG_SIZE, IMG_SIZE, 3), (BATCH_SIZE,)))
    val_dataset = tf.data.Dataset.from_generator(lambda: val_dataset, output_types=(tf.float32, tf.int32), output_shapes=((BATCH_SIZE, CLIP_LEN, IMG_SIZE, IMG_SIZE, 3), (BATCH_SIZE,)))

    num_classes = len(set(train_dataset.label_array))

    model = build_model(num_classes)
    compile_model(model)

    history = train_model(model, train_dataset, val_dataset)

    model.save('ssv2_model.h5')
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    main()