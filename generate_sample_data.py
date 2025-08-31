import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Output directory
output_dir = "data/raw/sample_1"
os.makedirs(output_dir, exist_ok=True)

# Create a simple 256x256 grid
x = np.linspace(-2, 2, 256)
y = np.linspace(-2, 2, 256)
x, y = np.meshgrid(x, y)

def create_sample_data(base_value, variation_scale, noise_scale=0.1):
    """Generate sample data with some spatial variation."""
    # Base pattern (hill in the center)
    z = base_value + variation_scale * np.exp(-(x**2 + y**2))
    # Add some noise
    z = z + noise_scale * np.random.randn(*x.shape)
    return z.astype(np.float32)

# Create sample data for each feature
features = {
    'elevation': (100, 50),         # meters
    'wind_speed': (5, 3),           # m/s
    'temperature': (25, 10),        # °C
    'humidity': (60, 20),           # %
    'precipitation': (5, 4),        # mm
    'vegetation_density': (0.5, 0.3),  # 0-1
    'fuel_moisture': (0.3, 0.2),    # 0-1
    'historical_burn_area': (0.1, 0.05),  # 0-1
    'fire_mask': (0, 0.5)           # Binary mask (0 or 1)
}

# Define the spatial reference and transform
crs = 'EPSG:4326'
transform = from_origin(-119.0, 38.0, 0.01, 0.01)  # Example coordinates

# Save each feature as a GeoTIFF
for name, (base, scale) in features.items():
    data = create_sample_data(base, scale)
    if name == 'fire_mask':
        # Make it a binary mask
        data = (data > 0.5).astype(np.uint8)
    
    with rasterio.open(
        os.path.join(output_dir, f"{name}.tif"),
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

print(f"Sample data generated in {output_dir}")
