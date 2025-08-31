import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Import from local modules using relative imports
from .model import get_model, get_callbacks
from .data_processing import load_and_preprocess_data, create_data_generator
from . import config

def plot_training_history(history, output_dir):
    """Plot training history metrics."""
    # Plot training & validation loss values
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Dice coefficient plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'])
    plt.plot(history.history['val_dice_coefficient'])
    plt.title('Model Dice Coefficient')
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to {plot_path}")

def train_model(data_dir, epochs=None, batch_size=None):
    """
    Train the wildfire prediction model.
    
    Args:
        data_dir: Directory containing the training data
        epochs: Number of training epochs (uses config if None)
        batch_size: Batch size (uses config if None)
    """
    # Use config values if not provided
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_val, y_train, y_val = load_and_preprocess_data(data_dir)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Input shape: {X_train[0].shape}")
    
    # Create data generators
    print("Creating data generators...")
    train_generator = create_data_generator(X_train, y_train, batch_size=batch_size, augment=True)
    val_generator = create_data_generator(X_val, y_val, batch_size=batch_size, augment=False)
    
    # Calculate steps per epoch
    train_steps = len(X_train) // batch_size
    val_steps = len(X_val) // batch_size
    
    # Create and compile model
    print("Creating model...")
    model = get_model()
    model.summary()
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        validation_data=val_generator,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    final_model_path = os.path.join(config.MODEL_DIR, 'final_model.h5')
    model.save(final_model_path)
    print(f"\nTraining complete. Final model saved to {final_model_path}")
    
    # Plot training history
    plot_training_history(history, config.OUTPUT_DIR)
    
    return history

def main():
    parser = argparse.ArgumentParser(description='Train a wildfire prediction model.')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing the training data')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Train the model
    history = train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
