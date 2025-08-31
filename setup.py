from setuptools import setup, find_packages

setup(
    name="wildfire_simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.6.0',
        'numpy>=1.19.5',
        'rasterio>=1.2.10',
        'scikit-learn>=0.24.2',
        'scikit-image>=0.18.3',
        'matplotlib>=3.4.3',
    ],
    python_requires='>=3.8',
)
