import os
import numpy as np
import rasterio
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------------
# TFRecord Parsing
# ----------------------------
def parse_tfrecord(example_proto, input_shape=(128, 128, 3), mask_shape=(128, 128, 1)):
    """
    Parse a single TFRecord example into image and mask.
    """
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string)
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    image = tf.io.decode_raw(parsed['image'], tf.float32)
    mask = tf.io.decode_raw(parsed['mask'], tf.float32)
    
    image = tf.reshape(image, input_shape)
    mask = tf.reshape(mask, mask_shape)
    
    return image, mask

# ----------------------------
# Load and preprocess TFRecords
# ----------------------------
def load_and_preprocess_data(data_dir, input_shape=(128, 128, 3), mask_shape=(128, 128, 1), test_size=0.2):
    """
    Load all training TFRecords and split into training and validation sets.
    """
    # List all training TFRecords
    tfrecord_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if "train" in f]
    if len(tfrecord_files) == 0:
        raise ValueError(f"No training TFRecords found in {data_dir}")
    
    # Create TFRecord dataset
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(lambda x: parse_tfrecord(x, input_shape, mask_shape))
    
    # Convert dataset to NumPy arrays
    images, masks = [], []
    for img, msk in dataset:
        images.append(img.numpy())
        masks.append(msk.numpy())
    
    X = np.array(images, dtype=np.float32)
    y = np.array(masks, dtype=np.float32)
    
    # Split into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"Loaded {len(X_train)} training samples and {len(X_val)} validation samples")
    
    return X_train, X_val, y_train, y_val

# ----------------------------
# Data Generator
# ----------------------------
def create_data_generator(X, y, batch_size=32, augment=False):
    """
    Simple data generator for training.
    """
    def generator():
        while True:
            indices = np.arange(len(X))
            if augment:
                np.random.shuffle(indices)
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                batch_X = X[batch_idx]
                batch_y = y[batch_idx]
                
                # Optional augmentation
                if augment:
                    for i in range(len(batch_X)):
                        if np.random.rand() > 0.5:
                            batch_X[i] = np.flip(batch_X[i], axis=1)
                            batch_y[i] = np.flip(batch_y[i], axis=1)
                
                yield batch_X, batch_y
    return generator()
