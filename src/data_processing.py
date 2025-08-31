import os
import numpy as np
import rasterio
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_raster(file_path):
    """Load a single raster file and return as numpy array."""
    with rasterio.open(file_path) as src:
        return src.read(1)

def load_sample(sample_dir):
    """Load all features for a single sample."""
    features = ['elevation', 'wind_speed', 'temperature', 'humidity', 
               'precipitation', 'vegetation_density', 'fuel_moisture', 'historical_burn_area']
    
    X = []
    for feature in features:
        file_path = os.path.join(sample_dir, f"{feature}.tif")
        if os.path.exists(file_path):
            arr = load_raster(file_path)
            X.append(arr)
    
    # Load target mask
    mask_path = os.path.join(sample_dir, 'fire_mask.tif')
    y = load_raster(mask_path) if os.path.exists(mask_path) else None
    
    return np.stack(X, axis=-1), y

def preprocess_data(X, y, target_size=(256, 256)):
    """Preprocess the data (resize and normalize)."""
    # Resize
    X_resized = np.array([resize(img, target_size, preserve_range=True) for img in X])
    y_resized = resize(y, target_size, preserve_range=True, order=0, preserve_binary=True)
    
    # Normalize features to [0, 1]
    X_norm = (X_resized - X_resized.min(axis=(1, 2), keepdims=True)) / \
             (X_resized.max(axis=(1, 2), keepdims=True) - X_resized.min(axis=(1, 2), keepdims=True) + 1e-8)
    
    # Ensure y is binary
    y_binary = (y_resized > 0.5).astype(np.float32)
    
    return X_norm, y_binary

def load_and_preprocess_data(data_dir):
    """Load and preprocess all samples."""
    samples = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    X_list, y_list = [], []
    for sample in samples:
        sample_dir = os.path.join(data_dir, sample)
        X, y = load_sample(sample_dir)
        if y is not None:
            X_list.append(X)
            y_list.append(y)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_list, y_list, test_size=0.2, random_state=42
    )
    
    return np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)

def create_data_generator(X, y, batch_size=8, augment=True):
    """Create a data generator with optional augmentation."""
    if augment:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator()
    
    # Add channel dimension if needed
    if len(X.shape) == 3:
        X = np.expand_dims(X, -1)
    if len(y.shape) == 3:
        y = np.expand_dims(y, -1)
    
    # Create generator
    seed = 42
    image_generator = datagen.flow(
        X, y,
        batch_size=batch_size,
        seed=seed
    )
    
    return image_generator
