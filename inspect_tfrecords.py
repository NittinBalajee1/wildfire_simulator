import os
import tensorflow as tf
from pathlib import Path

def print_tfrecord_structure(file_path):
    """Print the structure and statistics of a TFRecord file."""
    print(f"\nInspecting: {file_path}")
    
    raw_dataset = tf.data.TFRecordDataset(file_path)
    
    # Get the first record
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print("\nFeatures in the TFRecord:")
        for feature_name, feature in example.features.feature.items():
            values = getattr(feature, feature.WhichOneof('kind')).value
            print(f"\n- {feature_name}:")
            print(f"  Type: {feature.WhichOneof('kind')}")
            print(f"  Number of values: {len(values)}")
            
            # Calculate basic statistics for numerical features
            if feature.WhichOneof('kind') == 'float_list':
                values = list(map(float, values))
                print(f"  Min: {min(values):.4f}")
                print(f"  Max: {max(values):.4f}")
                print(f"  Mean: {sum(values)/len(values):.4f}")
                
                # Print first 5 values as sample
                print("  Sample values:")
                for v in values[:5]:
                    print(f"    {v:.4f}")
                if len(values) > 5:
                    print("    ...")

def main():
    # Define the directory containing TFRecord files
    data_dir = Path("data/raw/next-day-wildfire-spread")
    
    # Find all TFRecord files
    tfrecord_files = list(data_dir.glob("**/*.tfrecord*"))  # **/ searches in all subdirectories
    
    if not tfrecord_files:
        print(f"No TFRecord files found in {data_dir}")
        return
    
    print(f"Found {len(tfrecord_files)} TFRecord file(s):")
    for i, file in enumerate(tfrecord_files, 1):
        print(f"  {i}. {file}")
    
    # Process each TFRecord file
    for file in tfrecord_files:
        try:
            print_tfrecord_structure(file)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    main()