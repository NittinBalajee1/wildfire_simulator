import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Model paths
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'

# Model parameters
INPUT_SHAPE = (256, 256, 1)  # Input shape for the model (height, width, channels)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Data processing
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42

# Features to include from the dataset
FEATURES = [
    'elevation',
    'wind_speed',
    'wind_direction',
    'temperature',
    'humidity',
    'precipitation',
    'vegetation_density',
    'fuel_moisture',
    'historical_burn_area'
]

# Output classes
NUM_CLASSES = 2  # Binary classification: fire spread or no spread

# Training parameters
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# Map visualization
MAP_CENTER = [37.8, -119.4]  # Default center (California)
DEFAULT_ZOOM = 6

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / 'predictions', exist_ok=True)
os.makedirs(OUTPUT_DIR / 'visualizations', exist_ok=True)
