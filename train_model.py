import tensorflow as tf
from pathlib import Path
import numpy as np
from datetime import datetime
from tensorflow.keras import layers, models

def create_model(input_shape=(64, 64, 11)):
    """Create a U-Net like model for fire spread prediction."""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bottleneck
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.concatenate([up1, conv2], axis=-1)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(up1)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = layers.concatenate([up2, conv1], axis=-1)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(up2)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def parse_tfrecord(example_proto):
    """Parse the input tf.Example proto using the dictionary of features."""
    feature_description = {
        'tmmn': tf.io.FixedLenFeature([4096], tf.float32),
        'tmmx': tf.io.FixedLenFeature([4096], tf.float32),
        'NDVI': tf.io.FixedLenFeature([4096], tf.float32),
        'elevation': tf.io.FixedLenFeature([4096], tf.float32),
        'FireMask': tf.io.FixedLenFeature([4096], tf.float32),
        'PrevFireMask': tf.io.FixedLenFeature([4096], tf.float32),
        'pdsi': tf.io.FixedLenFeature([4096], tf.float32),
        'vs': tf.io.FixedLenFeature([4096], tf.float32),
        'pr': tf.io.FixedLenFeature([4096], tf.float32),
        'sph': tf.io.FixedLenFeature([4096], tf.float32),
        'th': tf.io.FixedLenFeature([4096], tf.float32),
        'erc': tf.io.FixedLenFeature([4096], tf.float32),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def prepare_dataset(record):
    """Prepare the dataset for training."""
    # Select features and reshape to (64, 64, 1)
    features = {
        'elevation': tf.reshape(record['elevation'], (64, 64, 1)),
        'NDVI': tf.reshape(record['NDVI'], (64, 64, 1)),
        'tmmx': tf.reshape(record['tmmx'], (64, 64, 1)),
        'tmmn': tf.reshape(record['tmmn'], (64, 64, 1)),
        'vs': tf.reshape(record['vs'], (64, 64, 1)),
        'pdsi': tf.reshape(record['pdsi'], (64, 64, 1)),
        'pr': tf.reshape(record['pr'], (64, 64, 1)),
        'sph': tf.reshape(record['sph'], (64, 64, 1)),
        'th': tf.reshape(record['th'], (64, 64, 1)),
        'erc': tf.reshape(record['erc'], (64, 64, 1)),
        'PrevFireMask': tf.reshape(record['PrevFireMask'], (64, 64, 1)),
    }
    
    # Stack features along the channel dimension
    x = tf.concat(list(features.values()), axis=-1)
    
    # Normalize features
    x = (x - tf.reduce_mean(x, axis=[0,1], keepdims=True)) / tf.math.reduce_std(x, axis=[0,1], keepdims=True)
    
    # Target is the FireMask
    y = tf.reshape(record['FireMask'], (64, 64, 1))
    
    return x, y

def main():
    # Set up paths
    data_dir = Path("data/raw/next-day-wildfire-spread")
    
    # Load and prepare training dataset
    train_files = list(data_dir.glob("*train*.tfrecord*"))
    train_dataset = tf.data.TFRecordDataset(train_files)
    train_dataset = train_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.map(prepare_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Load and prepare validation dataset
    val_files = list(data_dir.glob("*eval*.tfrecord*"))
    val_dataset = tf.data.TFRecordDataset(val_files)
    val_dataset = val_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(prepare_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    batch_size = 16
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create and compile model
    model = create_model()
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    
    # Set up callbacks
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    print("Starting model training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[tensorboard_callback, early_stopping]
    )
    
    # Save the model
    model.save('wildfire_spread_model.h5')
    print("\nTraining complete! Model saved as 'wildfire_spread_model.h5'")
    print(f"TensorBoard logs saved in {log_dir}")

if __name__ == "__main__":
    main()
