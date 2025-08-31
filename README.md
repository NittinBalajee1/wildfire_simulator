# Wildfire Propagation Simulator

An AI-powered system for predicting wildfire spread using deep learning and geospatial data.

## Features

- Predicts wildfire spread using a U-Net based deep learning model
- Processes multiple environmental factors (elevation, wind, temperature, etc.)
- Generates visualizations and GeoTIFF outputs
- Includes data preprocessing and augmentation

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd wildfire_simulator
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
wildfire_simulator/
├── data/                   # Data directory
│   ├── raw/               # Raw input data
│   └── processed/         # Processed data
├── models/                # Saved models
├── notebooks/             # Jupyter notebooks for exploration
├── outputs/               # Prediction outputs
│   ├── predictions/       # Prediction results
│   └── visualizations/    # Visualization outputs
└── src/                   # Source code
    ├── config.py          # Configuration settings
    ├── data_processing.py # Data loading and preprocessing
    ├── model.py          # Model architecture
    ├── predict.py        # Prediction script
    └── train.py          # Training script
```

## Getting Started

### Data Preparation

1. Organize your data in the following structure:
   ```
   data/raw/
   ├── sample_1/
   │   ├── elevation.tif
   │   ├── wind_speed.tif
   │   ├── temperature.tif
   │   ├── humidity.tif
   │   ├── precipitation.tif
   │   ├── vegetation_density.tif
   │   ├── fuel_moisture.tif
   │   ├── historical_burn_area.tif
   │   └── fire_mask.tif
   └── sample_2/
       └── ...
   ```

### Training the Model

```bash
python -m src.train --data-dir data/raw --epochs 50 --batch-size 32
```

### Making Predictions

```bash
python -m src.predict --model-path models/best_model.h5 --sample-dir data/raw/sample_1 --output-dir outputs/predictions
```

## Configuration

Modify `src/config.py` to adjust model parameters, data paths, and training settings.

## Dependencies

- Python 3.8+
- TensorFlow 2.6+
- Rasterio
- NumPy
- Matplotlib
- scikit-image
- scikit-learn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
