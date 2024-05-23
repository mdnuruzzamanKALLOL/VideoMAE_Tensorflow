import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
CLIP_LEN = 8
FRAME_SAMPLE_RATE = 2

# Data Augmentation
def preprocess(video, label, training=True):
    # Resize and normalize the video frames
    video = tf.image.resize(video, (IMG_SIZE, IMG_SIZE))
    video = tf.image.random_flip_left_right(video) if training else video
    video = tf.image.random_flip_up_down(video) if training else video
    video = tf.clip_by_value(video, 0, 255) / 255.0
    return video, label

# Load the Kinetics dataset
def load_kinetics_dataset(mode='train'):
    dataset, info = tfds.load("kinetics700", split=mode, with_info=True, as_supervised=True)
    dataset = dataset.map(lambda x, y: preprocess(x, y, training=(mode=='train')))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, info

# Define the model
def build_model(num_classes):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Compile the model
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Train the model
def train_model(model, train_dataset, val_dataset):
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset
    )
    return history

# Main function
def main():
    train_dataset, info = load_kinetics_dataset('train')
    val_dataset, _ = load_kinetics_dataset('validation')
    num_classes = info.features['label'].num_classes
    
    model = build_model(num_classes)
    compile_model(model)
    
    history = train_model(model, train_dataset, val_dataset)
    
    model.save('kinetics_model.h5')
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    main()
