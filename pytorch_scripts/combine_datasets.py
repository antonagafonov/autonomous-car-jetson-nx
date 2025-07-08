import numpy as np
import warnings

# Suppress the FutureWarning about np.bool
warnings.filterwarnings('ignore', category=FutureWarning)

# Fix numpy compatibility issue - must be done before pandas import
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int  
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str

import pandas as pd
import os
from pathlib import Path

def combine_car_datasets(dataset1_path, dataset2_path, output_path):
    """
    Combine two car behavior datasets with proper image path handling.
    
    Args:
        dataset1_path: Path to first dataset CSV
        dataset2_path: Path to second dataset CSV  
        output_path: Path where to save the combined dataset
    """
    
    # Load both datasets
    print("Loading dataset 1...")
    df1 = pd.read_csv(dataset1_path)
    print(f"Dataset 1 shape: {df1.shape}")
    
    print("Loading dataset 2...")
    df2 = pd.read_csv(dataset2_path)
    print(f"Dataset 2 shape: {df2.shape}")
    
    # Get the base directory paths (parent of 'data' directory)
    dataset1_base_dir = os.path.dirname(os.path.dirname(dataset1_path))  # Go up from data/ to base/
    dataset2_base_dir = os.path.dirname(os.path.dirname(dataset2_path))  # Go up from data/ to base/
    
    # Construct image directory paths
    dataset1_images_dir = os.path.join(dataset1_base_dir, 'images')
    dataset2_images_dir = os.path.join(dataset2_base_dir, 'images')
    
    # Update image paths to include full directory structure
    print("Updating image paths...")
    print(f"Dataset 1 images directory: {dataset1_images_dir}")
    print(f"Dataset 2 images directory: {dataset2_images_dir}")
    
    # For dataset 1: prepend the images directory path
    df1_copy = df1.copy()
    df1_copy['image_filename'] = df1_copy['image_filename'].apply(
        lambda x: os.path.join(dataset1_images_dir, x)
    )
    
    # For dataset 2: prepend the images directory path  
    df2_copy = df2.copy()
    df2_copy['image_filename'] = df2_copy['image_filename'].apply(
        lambda x: os.path.join(dataset2_images_dir, x)
    )
    
    # Add dataset source identifier for tracking
    df1_copy['dataset_source'] = 'behavior_20250703_183901'
    df2_copy['dataset_source'] = 'behavior_20250707_085252'
    
    # Combine the datasets
    print("Combining datasets...")
    combined_df = pd.concat([df1_copy, df2_copy], ignore_index=True)
    
    # Reset image sequence numbers to be continuous
    combined_df['image_seq'] = range(len(combined_df))
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Total samples: {len(combined_df)}")
    
    # Show distribution by dataset source
    print("\nDataset distribution:")
    try:
        print(combined_df['dataset_source'].value_counts())
    except Exception as e:
        print(f"Could not display dataset source distribution: {e}")
        # Fallback to manual counting
        source_counts = {}
        for source in combined_df['dataset_source']:
            source_counts[source] = source_counts.get(source, 0) + 1
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
    
    # Show distribution by category if available
    if 'manual_category' in combined_df.columns:
        print("\nCategory distribution:")
        try:
            print(combined_df['manual_category'].value_counts())
        except Exception as e:
            print(f"Could not display category distribution: {e}")
    
    # Save the combined dataset
    print(f"\nSaving combined dataset to: {output_path}")
    combined_df.to_csv(output_path, index=False)
    
    # Verify image paths exist
    print("\nVerifying image paths...")
    missing_images = []
    sample_size = min(100, len(combined_df))  # Check first 100 images
    
    for idx, row in combined_df.head(sample_size).iterrows():
        img_path = row['image_filename']
        if not os.path.exists(img_path):
            missing_images.append(img_path)
    
    if missing_images:
        print(f"Warning: {len(missing_images)} out of {sample_size} checked images are missing!")
        print("First few missing files:")
        for img in missing_images[:5]:
            print(f"  - {img}")
        print("Expected path format: /home/toon/car_datasets/behavior_YYYYMMDD_HHMMSS_extracted/images/image_XXXXXX.png")
    else:
        print(f"✓ All {sample_size} checked image paths exist!")
        print(f"Sample image path: {combined_df.iloc[0]['image_filename']}")
    
    return combined_df

# Usage example
if __name__ == "__main__":
    # Define your dataset paths
    dataset1_csv = "/home/toon/car_datasets/behavior_20250703_183901_extracted/data/balanced_synchronized_dataset.csv"
    dataset2_csv = "/home/toon/car_datasets/behavior_20250707_085252_extracted/data/balanced_synchronized_dataset.csv"
    
    # Output path for combined dataset
    output_csv = "/home/toon/car_datasets/combined_balanced_synchronized_dataset.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Combine the datasets
    combined_data = combine_car_datasets(dataset1_csv, dataset2_csv, output_csv)
    
    print(f"\n✓ Successfully combined datasets!")
    print(f"Combined dataset saved to: {output_csv}")
    print(f"Combined dataset contains {len(combined_data)} samples")
    
    # Display some sample rows
    print("\nSample rows from combined dataset:")
    sample_cols = ['image_filename', 'dataset_source', 'manual_category'] if 'manual_category' in combined_data.columns else ['image_filename', 'dataset_source']
    print(combined_data[sample_cols].head())
    
    # Show expected vs actual path format
    print(f"\nExpected image path format:")
    print(f"  /home/toon/car_datasets/behavior_YYYYMMDD_HHMMSS_extracted/images/image_XXXXXX.png")
    print(f"Actual first image path:")
    print(f"  {combined_data.iloc[0]['image_filename']}")