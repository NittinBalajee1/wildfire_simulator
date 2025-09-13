import tensorflow as tf
import os

def _parse_tfrecord(example_proto):
    # Define your TFRecord feature description
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    image = tf.io.decode_raw(parsed['image'], tf.float32)
    mask = tf.io.decode_raw(parsed['mask'], tf.float32)
    
    # reshape according to your image dimensions
    image = tf.reshape(image, [128, 128, 3])
    mask = tf.reshape(mask, [128, 128, 1])
    
    return image, mask

def load_and_preprocess_data(data_dir):
    tfrecord_files = tf.io.gfile.glob(os.path.join(data_dir, "*.tfrecord"))
    if len(tfrecord_files) == 0:
        raise ValueError(f"No TFRecord files found in {data_dir}")
    
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(_parse_tfrecord)
    
    X = []
    y = []
    for img, mask in dataset:
        X.append(img.numpy())
        y.append(mask.numpy())
    
    # Split into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val
