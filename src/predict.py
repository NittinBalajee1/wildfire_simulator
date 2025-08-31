import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from tensorflow.keras.models import load_model
from . import config
from .model import dice_coefficient, dice_loss

def load_trained_model(model_path):
    """Load a trained model with custom objects."""
    return load_model(model_path, custom_objects={
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss
    }, compile=False)

def predict_fire_spread(model, sample_dir, output_dir):
    """Make and save predictions for a sample."""
    # Prepare input
    input_data = np.random.rand(1, *config.INPUT_SHAPE)  # Replace with actual data loading
    
    # Predict
    prediction = model.predict(input_data)
    binary_mask = (prediction > 0.5).astype(np.uint8)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualization
    plt.figure(figsize=(10, 5))
    plt.imshow(binary_mask[0, :, :, 0], cmap='Reds')
    plt.title('Predicted Fire Spread')
    plt.axis('off')
    plt.savefig(f"{output_dir}/prediction.png", bbox_inches='tight')
    plt.close()
    
    # Save as GeoTIFF
    with rasterio.open(f"{sample_dir}/elevation.tif") as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(f"{output_dir}/prediction.tif", 'w', **profile) as dst:
            dst.write(binary_mask[0, :, :, 0], 1)
    
    return binary_mask

def main():
    parser = argparse.ArgumentParser(description='Predict wildfire spread.')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--sample-dir', required=True, help='Directory with input data')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    
    print("Loading model...")
    model = load_trained_model(args.model_path)
    
    print("Making predictions...")
    predict_fire_spread(model, args.sample_dir, args.output_dir)
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    import argparse
    main()
