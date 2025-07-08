#!/usr/bin/env python3
"""
Data Balancing Script for Manual Control Values
Balances dataset based on sum of absolute values in manual_angular_z_next_30 sequences
Categories: high_angular_activity (sum >= 10) vs low_angular_activity (sum < 10)
Includes future prediction columns (next_10, next_30, next_60) and option to remove initial samples
"""

# Fix numpy compatibility BEFORE importing pandas
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

# Set matplotlib to use non-interactive backend to avoid display issues
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

# Now safe to import pandas and other modules
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import argparse
import sys

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Balance dataset based on sum of abs(manual_angular_z_next_30) values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python balance_data.py data/synchronized_dataset.csv
  python balance_data.py /path/to/dataset.csv --balance-type equal
  python balance_data.py dataset.csv --output-dir ./balanced_output
  python balance_data.py dataset.csv --seed 123
  python balance_data.py dataset.csv --balance-type reduce_zero_angular --reduce-angular-zero-ratio 0.3
  python balance_data.py dataset.csv --remove-first-samples 500

Balancing Criteria:
  - Categories are created based on sum of absolute values in manual_angular_z_next_30
  - high_angular_activity: sum >= 10
  - low_angular_activity: sum < 10
        '''
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',  # Make it optional so we can provide a default
        help='Path to the input CSV file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for balanced dataset and plots (default: same as input file directory)',
        default=None
    )
    
    parser.add_argument(
        '--balance-type', '-b',
        choices=['equal', 'custom', 'reduce_zero_angular'],
        default='reduce_zero_angular',
        help='Type of balancing: equal (balance high/low activity), reduce_zero_angular (reduce low activity), custom (default: equal)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=30.0,
        help='Threshold for sum of abs(manual_angular_z_next_30) to categorize as high activity (default: 30.0)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducible sampling (default: 42)'
    )
    

    parser.add_argument(
        '--prefix', '-p',
        default='balanced_',
        help='Prefix for output files (default: balanced_)'
    )
    
    parser.add_argument(
        '--reduce-angular-zero-ratio', '-r',
        type=float,
        default=0.5,
        help='When using reduce_zero_angular, ratio to reduce low angular activity samples (default: 0.5 = 50%% reduction)'
    )
    
    parser.add_argument(
        '--remove-first-samples', '-n',
        type=int,
        default=300,
        help='Number of first samples to remove from dataset (default: 300)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    return parser.parse_args()

def load_and_analyze_data(file_path):
    """Load CSV data and analyze manual control values"""
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for manual control columns
    manual_cols = [col for col in df.columns if 'manual' in col]
    print(f"Manual control columns found: {manual_cols}")
    
    # Verify required columns exist
    if 'manual_linear_x' not in df.columns:
        print("Error: manual_linear_x column not found in dataset!")
        sys.exit(1)
    if 'manual_angular_z' not in df.columns:
        print("Error: manual_angular_z column not found in dataset!")
        sys.exit(1)
    
    return df

def add_future_columns(df):
    """Add future prediction columns (sequences of next_10, next_30, and next_60 samples)"""
    print("\nAdding future prediction columns...")
    
    # Convert manual columns to numeric
    df['manual_linear_x'] = pd.to_numeric(df['manual_linear_x'], errors='coerce')
    df['manual_angular_z'] = pd.to_numeric(df['manual_angular_z'], errors='coerce')
    
    # Handle NaN values
    df['manual_linear_x'].fillna(0.0, inplace=True)
    df['manual_angular_z'].fillna(0.0, inplace=True)
    
    # Create arrays of future values for each row
    print("Creating future sequences (this may take a moment)...")
    
    linear_x_values = df['manual_linear_x'].values
    angular_z_values = df['manual_angular_z'].values
    
    # Initialize lists to store future sequences
    linear_x_next_10 = []
    angular_z_next_10 = []
    linear_x_next_30 = []
    angular_z_next_30 = []
    linear_x_next_60 = []
    angular_z_next_60 = []
    
    for i in range(len(df)):
        # Get next 10 values for linear_x
        if i + 10 < len(linear_x_values):
            next_10_linear = linear_x_values[i+1:i+11].tolist()
        else:
            # Pad with zeros if not enough future values
            available = linear_x_values[i+1:].tolist()
            padding = [0.0] * (10 - len(available))
            next_10_linear = available + padding
        
        # Get next 10 values for angular_z
        if i + 10 < len(angular_z_values):
            next_10_angular = angular_z_values[i+1:i+11].tolist()
        else:
            # Pad with zeros if not enough future values
            available = angular_z_values[i+1:].tolist()
            padding = [0.0] * (10 - len(available))
            next_10_angular = available + padding
        
        # Get next 30 values for linear_x
        if i + 30 < len(linear_x_values):
            next_30_linear = linear_x_values[i+1:i+31].tolist()
        else:
            # Pad with zeros if not enough future values
            available = linear_x_values[i+1:].tolist()
            padding = [0.0] * (30 - len(available))
            next_30_linear = available + padding
        
        # Get next 60 values for linear_x
        if i + 60 < len(linear_x_values):
            next_60_linear = linear_x_values[i+1:i+61].tolist()
        else:
            # Pad with zeros if not enough future values
            available = linear_x_values[i+1:].tolist()
            padding = [0.0] * (60 - len(available))
            next_60_linear = available + padding
        
        # Get next 30 values for angular_z
        if i + 30 < len(angular_z_values):
            next_30_angular = angular_z_values[i+1:i+31].tolist()
        else:
            # Pad with zeros if not enough future values
            available = angular_z_values[i+1:].tolist()
            padding = [0.0] * (30 - len(available))
            next_30_angular = available + padding
        
        # Get next 60 values for angular_z
        if i + 60 < len(angular_z_values):
            next_60_angular = angular_z_values[i+1:i+61].tolist()
        else:
            # Pad with zeros if not enough future values
            available = angular_z_values[i+1:].tolist()
            padding = [0.0] * (60 - len(available))
            next_60_angular = available + padding
        
        # Store as JSON strings for CSV compatibility
        linear_x_next_10.append(str(next_10_linear))
        angular_z_next_10.append(str(next_10_angular))
        linear_x_next_30.append(str(next_30_linear))
        linear_x_next_60.append(str(next_60_linear))
        angular_z_next_30.append(str(next_30_angular))
        angular_z_next_60.append(str(next_60_angular))
    
    # Add the columns to dataframe
    df['manual_linear_x_next_10'] = linear_x_next_10
    df['manual_angular_z_next_10'] = angular_z_next_10
    df['manual_linear_x_next_30'] = linear_x_next_30
    df['manual_linear_x_next_60'] = linear_x_next_60
    df['manual_angular_z_next_30'] = angular_z_next_30
    df['manual_angular_z_next_60'] = angular_z_next_60
    
    print(f"Added future prediction columns:")
    print(f"  - manual_linear_x_next_10 (contains 10 future values each)")
    print(f"  - manual_angular_z_next_10 (contains 10 future values each)")
    print(f"  - manual_linear_x_next_30 (contains 30 future values each)")
    print(f"  - manual_linear_x_next_60 (contains 60 future values each)") 
    print(f"  - manual_angular_z_next_30 (contains 30 future values each)")
    print(f"  - manual_angular_z_next_60 (contains 60 future values each)")
    print(f"  Note: Future values are stored as JSON-like strings for CSV compatibility")
    
    return df

def round_float_columns(df, decimal_places=4):
    """Round all float columns to specified decimal places"""
    print(f"\nRounding float columns to {decimal_places} decimal places...")
    
    # Use string-based dtype selection to avoid numpy compatibility issues
    try:
        float_columns = df.select_dtypes(include=['float64', 'float32', 'float']).columns
        print(f"Float columns found: {list(float_columns)}")
        
        for col in float_columns:
            df[col] = df[col].round(decimal_places)
    except Exception as e:
        print(f"Warning: Could not auto-detect float columns: {e}")
        print("Falling back to manual detection...")
        
        # Manual detection of float columns
        float_columns = []
        for col in df.columns:
            if df[col].dtype.kind in 'f':  # 'f' represents floating point types
                float_columns.append(col)
        
        print(f"Float columns found (manual detection): {float_columns}")
        for col in float_columns:
            df[col] = df[col].round(decimal_places)
    
    # Round specific manual columns (but skip the future sequence columns as they're now strings)
    manual_columns = ['manual_linear_x', 'manual_angular_z']
    
    for col in manual_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(decimal_places)
            df[col].fillna(0.0, inplace=True)
    
    # Handle future sequence columns separately (they contain arrays as strings)
    future_columns = ['manual_linear_x_next_10', 'manual_angular_z_next_10',
                     'manual_linear_x_next_30', 'manual_linear_x_next_60',
                     'manual_angular_z_next_30', 'manual_angular_z_next_60']
    
    for col in future_columns:
        if col in df.columns:
            print(f"Processing future sequence column: {col}")
            # Parse the string arrays, round the values, and convert back to strings
            rounded_sequences = []
            for seq_str in df[col]:
                try:
                    # Parse the string representation of the list
                    seq = eval(seq_str)  # Convert string to list
                    # Round each value in the sequence
                    rounded_seq = [round(float(val), decimal_places) for val in seq]
                    rounded_sequences.append(str(rounded_seq))
                except:
                    # If parsing fails, keep original
                    rounded_sequences.append(seq_str)
            df[col] = rounded_sequences
    
    return df

def remove_first_samples(df, n_samples):
    """Remove first N samples from the dataset"""
    if n_samples <= 0:
        print("No samples to remove (n_samples <= 0)")
        return df
    
    original_size = len(df)
    if n_samples >= original_size:
        print(f"Warning: Requested to remove {n_samples} samples, but dataset only has {original_size} samples!")
        print("Removing all but the last 10 samples to avoid empty dataset")
        n_samples = max(0, original_size - 10)
    
    df_trimmed = df.iloc[n_samples:].reset_index(drop=True)
    
    print(f"\nRemoved first {n_samples} samples:")
    print(f"  Original size: {original_size}")
    print(f"  New size: {len(df_trimmed)}")
    print(f"  Removed: {original_size - len(df_trimmed)} samples")
    
    return df_trimmed

def calculate_sequence_sum_abs(sequence_str):
    """Calculate sum of absolute values from a sequence string"""
    try:
        # Parse the string representation of the list
        sequence = eval(sequence_str)
        # Calculate sum of absolute values
        return sum(abs(float(val)) for val in sequence)
    except:
        # If parsing fails, return 0
        return 0.0

def analyze_manual_distribution(df, threshold=10.0):
    """Analyze the distribution based on sum of abs(manual_angular_z_next_30)"""
    
    # Convert manual_linear_x and manual_angular_z to numeric, handling any non-numeric values
    df['manual_linear_x'] = pd.to_numeric(df['manual_linear_x'], errors='coerce')
    df['manual_angular_z'] = pd.to_numeric(df['manual_angular_z'], errors='coerce')
    
    # Handle NaN values
    df['manual_linear_x'].fillna(0.0, inplace=True)
    df['manual_angular_z'].fillna(0.0, inplace=True)
    
    print("\nCalculating sum of absolute values for manual_angular_z_next_30...")
    
    # Calculate sum of absolute values for each manual_angular_z_next_30 sequence
    angular_z_next_30_sums = []
    for i, sequence_str in enumerate(df['manual_angular_z_next_30']):
        sum_abs = calculate_sequence_sum_abs(sequence_str)
        angular_z_next_30_sums.append(sum_abs)
        
        # Print progress every 1000 rows
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(df)} rows...")
    
    df['angular_z_next_30_sum_abs'] = angular_z_next_30_sums
    
    # Create categories based on whether sum of abs(manual_angular_z_next_30) >= threshold
    print(f"\nCategorizing based on sum of abs(manual_angular_z_next_30) >= {threshold}...")
    
    categories = []
    high_angular_count = 0
    low_angular_count = 0
    
    for sum_abs in angular_z_next_30_sums:
        if sum_abs >= threshold:
            categories.append('high_angular_activity')
            high_angular_count += 1
        else:
            categories.append('low_angular_activity')
            low_angular_count += 1
    
    df['manual_category'] = categories
    
    # Count distribution manually to avoid numpy.object issues
    category_counts = Counter(categories)
    
    print(f"\nDistribution after preprocessing:")
    print(f"  high_angular_activity (sum_abs >= {threshold}): {high_angular_count} ({high_angular_count/len(df)*100:.1f}%)")
    print(f"  low_angular_activity (sum_abs < {threshold}): {low_angular_count} ({low_angular_count/len(df)*100:.1f}%)")
    
    # Show some statistics about the sums
    sum_stats = pd.Series(angular_z_next_30_sums).describe()
    print(f"\nSum of abs(manual_angular_z_next_30) statistics:")
    print(f"  Mean: {sum_stats['mean']:.4f}")
    print(f"  Std: {sum_stats['std']:.4f}")
    print(f"  Min: {sum_stats['min']:.4f}")
    print(f"  Max: {sum_stats['max']:.4f}")
    print(f"  25%: {sum_stats['25%']:.4f}")
    print(f"  50%: {sum_stats['50%']:.4f}")
    print(f"  75%: {sum_stats['75%']:.4f}")
    
    return df, category_counts

def create_before_plots(df, category_counts, threshold=10.0):
    """Create histograms showing distribution before balancing"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Manual angular z distribution
    axes[0, 0].hist(df['manual_angular_z'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Manual Angular Z Distribution (Before Balancing)')
    axes[0, 0].set_xlabel('Manual Angular Z Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sum of abs(manual_angular_z_next_30) distribution
    axes[0, 1].hist(df['angular_z_next_30_sum_abs'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Sum of Abs(Angular Z Next 30) Distribution (Before Balancing)')
    axes[0, 1].set_xlabel('Sum of Absolute Values')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Category distribution
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    colors = ['lightcoral' if 'low' in cat else 'lightgreen' for cat in categories]
    axes[1, 0].bar(categories, counts, alpha=0.7, color=colors)
    axes[1, 0].set_title('Category Distribution (Before Balancing)')
    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot of manual_angular_z vs sum of abs values
    colors_scatter = ['red' if cat == 'low_angular_activity' else 'green' for cat in df['manual_category']]
    axes[1, 1].scatter(df['manual_angular_z'], df['angular_z_next_30_sum_abs'], 
                      alpha=0.5, s=1, c=colors_scatter)
    axes[1, 1].set_title('Manual Angular Z vs Sum Abs(Next 30) (Before Balancing)')
    axes[1, 1].set_xlabel('Manual Angular Z')
    axes[1, 1].set_ylabel('Sum of Abs(Angular Z Next 30)')
    axes[1, 1].axhline(y=threshold, color='blue', linestyle='--', label=f'Threshold ({threshold})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def balance_data(df, target_balance='equal', random_seed=42, reduce_angular_zero_ratio=0.5):
    """Balance the dataset based on sum of abs(manual_angular_z_next_30) categories"""
    
    # Count categories manually to avoid numpy.object issues
    category_counts = Counter(df['manual_category'].tolist())
    
    print(f"\nCategory counts before balancing:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    if target_balance == 'equal':
        # Balance to have equal numbers of high and low angular activity
        available_categories = list(category_counts.keys())
        
        if len(available_categories) < 2:
            print("Warning: Not enough categories for balancing!")
            return df
        
        min_count = min(category_counts.values())
        target_count = min_count
        
        print(f"\nBalancing to {target_count} samples per category...")
        
        balanced_dfs = []
        
        for category in ['high_angular_activity', 'low_angular_activity']:
            if category in category_counts.keys():
                category_df = df[df['manual_category'] == category]
                if len(category_df) >= target_count:
                    # Randomly sample target_count rows
                    sampled_df = category_df.sample(n=target_count, random_state=random_seed)
                    balanced_dfs.append(sampled_df)
                    print(f"  {category}: {len(category_df)} -> {target_count}")
                else:
                    # Keep all rows if less than target
                    balanced_dfs.append(category_df)
                    print(f"  {category}: {len(category_df)} (kept all)")
        
        if balanced_dfs:
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        else:
            print("Warning: No data remaining after balancing!")
            balanced_df = df
        
    elif target_balance == 'reduce_zero_angular':
        # Reduce low angular activity samples
        print(f"\nReducing low angular activity samples by ratio: {reduce_angular_zero_ratio}")
        
        balanced_dfs = []
        
        # Keep all high angular activity samples
        if 'high_angular_activity' in category_counts.keys():
            high_activity_df = df[df['manual_category'] == 'high_angular_activity']
            balanced_dfs.append(high_activity_df)
            print(f"  high_angular_activity: {len(high_activity_df)} (kept all)")
        
        # Reduce low angular activity samples
        if 'low_angular_activity' in category_counts.keys():
            low_activity_df = df[df['manual_category'] == 'low_angular_activity']
            original_count = len(low_activity_df)
            
            # Calculate target count based on reduction ratio
            target_count = int(original_count * reduce_angular_zero_ratio)
            
            if target_count > 0:
                # Randomly sample reduced number of rows
                sampled_df = low_activity_df.sample(n=target_count, random_state=random_seed)
                balanced_dfs.append(sampled_df)
                print(f"  low_angular_activity: {original_count} -> {target_count} (reduced by {(1-reduce_angular_zero_ratio)*100:.1f}%)")
            else:
                print(f"  low_angular_activity: {original_count} -> 0 (completely removed)")
        
        if balanced_dfs:
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        else:
            print("Warning: No data remaining after balancing!")
            balanced_df = df
            
    elif target_balance == 'custom':
        # Custom balancing logic can be added here
        print("Custom balancing not implemented yet, using original data")
        balanced_df = df
    else:
        balanced_df = df
    
    return balanced_df

def create_after_plots(balanced_df, threshold=10.0):
    """Create histograms showing distribution after balancing"""
    category_counts = Counter(balanced_df['manual_category'].tolist())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Manual angular z distribution
    axes[0, 0].hist(balanced_df['manual_angular_z'], bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0, 0].set_title('Manual Angular Z Distribution (After Balancing)')
    axes[0, 0].set_xlabel('Manual Angular Z Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sum of abs(manual_angular_z_next_30) distribution
    axes[0, 1].hist(balanced_df['angular_z_next_30_sum_abs'], bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0, 1].set_title('Sum of Abs(Angular Z Next 30) Distribution (After Balancing)')
    axes[0, 1].set_xlabel('Sum of Absolute Values')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Category distribution
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    colors = ['lightcoral' if 'low' in cat else 'lightgreen' for cat in categories]
    axes[1, 0].bar(categories, counts, alpha=0.7, color=colors)
    axes[1, 0].set_title('Category Distribution (After Balancing)')
    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot of manual_angular_z vs sum of abs values
    colors_scatter = ['red' if cat == 'low_angular_activity' else 'green' for cat in balanced_df['manual_category']]
    axes[1, 1].scatter(balanced_df['manual_angular_z'], balanced_df['angular_z_next_30_sum_abs'], 
                      alpha=0.5, s=1, c=colors_scatter)
    axes[1, 1].set_title('Manual Angular Z vs Sum Abs(Next 30) (After Balancing)')
    axes[1, 1].set_xlabel('Manual Angular Z')
    axes[1, 1].set_ylabel('Sum of Abs(Angular Z Next 30)')
    axes[1, 1].axhline(y=threshold, color='blue', linestyle='--', label=f'Threshold ({threshold})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # If no input file provided, use the default one
    if args.input_file is None:
        args.input_file = "/home/toon/car_datasets/behavior_20250703_183901_extracted/data/synchronized_dataset.csv"
        print(f"No input file specified, using default: {args.input_file}")
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        sys.exit(1)
    
    file_path = args.input_file
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(file_path) or '.'
    
    print(f"Input file: {file_path}")
    print(f"Output directory: {output_dir}")
    print(f"Balance type: {args.balance_type}")
    if args.balance_type == 'reduce_zero_angular':
        print(f"Zero angular reduction ratio: {args.reduce_angular_zero_ratio}")
    print(f"Remove first samples: {args.remove_first_samples}")
    print(f"Threshold: {args.threshold}")
    print(f"Random seed: {args.seed}")
    print(f"Output prefix: {args.prefix}")
    print("-" * 50)
    
    # Load data
    df = load_and_analyze_data(file_path)
    
    # Add future prediction columns
    df = add_future_columns(df)
    
    # Round all float columns to 4 decimal places
    df = round_float_columns(df, decimal_places=4)
    # save this df to same dire with rounded suffix
    name, ext = os.path.splitext(os.path.basename(file_path))
    rounded_filename = os.path.join(output_dir, f'{name}_rounded{ext}')
    df.to_csv(rounded_filename, index=False)
    print(f"Rounded dataset saved to: {rounded_filename}")

    # Remove first N samples
    df = remove_first_samples(df, args.remove_first_samples)
    
    # Analyze distribution after preprocessing
    df, preprocessed_counts = analyze_manual_distribution(df, threshold=args.threshold)
    # save preprocessed df
    preprocessed_filename = os.path.join(output_dir, f'{args.prefix}preprocessed{ext}')
    df.to_csv(preprocessed_filename, index=False)
    print(f"Preprocessed dataset saved to: {preprocessed_filename}")

    if not args.no_plots:
        # Create before plots
        print("\nCreating before balancing plots...")
        try:
            fig_before = create_before_plots(df, preprocessed_counts, threshold=args.threshold)
            
            # Save before plot
            before_plot_path = os.path.join(output_dir, f'{args.prefix}distribution_before.png')
            fig_before.savefig(before_plot_path, dpi=300, bbox_inches='tight')
            print(f"Before plots saved to: {before_plot_path}")
            plt.close(fig_before)
        except Exception as e:
            print(f"Warning: Could not create before plots: {e}")
    
    # Balance the data
    print("\nBalancing data...")
    balanced_df = balance_data(df, target_balance=args.balance_type, random_seed=args.seed, 
                              reduce_angular_zero_ratio=args.reduce_angular_zero_ratio)
    
    # Analyze balanced data
    print("\nBalanced distribution:")
    balanced_counts = Counter(balanced_df['manual_category'].tolist())
    for category, count in balanced_counts.items():
        print(f"  {category}: {count} ({count/len(balanced_df)*100:.1f}%)")
    
    if not args.no_plots:
        # Create after plots
        print("\nCreating after balancing plots...")
        try:
            fig_after = create_after_plots(balanced_df, threshold=args.threshold)
            
            # Save after plot
            after_plot_path = os.path.join(output_dir, f'{args.prefix}distribution_after.png')
            fig_after.savefig(after_plot_path, dpi=300, bbox_inches='tight')
            print(f"After plots saved to: {after_plot_path}")
            plt.close(fig_after)
        except Exception as e:
            print(f"Warning: Could not create after plots: {e}")
    
    # Save balanced dataset
    input_filename = os.path.basename(file_path)
    name, ext = os.path.splitext(input_filename)
    balanced_filename = os.path.join(output_dir, f'{args.prefix}{name}{ext}')
    
    # Remove the temporary columns before saving
    # columns_to_remove = ['manual_category', 'angular_z_next_30_sum_abs']
    # balanced_df_clean = balanced_df.drop(columns=[col for col in columns_to_remove if col in balanced_df.columns])
    # remove all rows that has low_angular_activity in manual_category
    balanced_df_clean = balanced_df
    balanced_df_clean.to_csv(balanced_filename, index=False)
    
    print(f"\nBalanced dataset saved to: {balanced_filename}")
    print(f"Dataset size after removing first {args.remove_first_samples} samples: {len(df)}")
    print(f"Final balanced dataset size: {len(balanced_df_clean)}")
    
    # Show summary statistics
    print("\nSummary Statistics:")
    print("Processed manual_angular_z stats:")
    try:
        print(df['manual_angular_z'].describe())
    except Exception as e:
        print(f"Could not display stats: {e}")
    
    print("\nBalanced manual_angular_z stats:")
    try:
        print(balanced_df['manual_angular_z'].describe())
    except Exception as e:
        print(f"Could not display stats: {e}")
    
    print("\nSum of abs(manual_angular_z_next_30) stats (processed):")
    try:
        print(df['angular_z_next_30_sum_abs'].describe())
    except Exception as e:
        print(f"Could not display stats: {e}")
        
    print("\nSum of abs(manual_angular_z_next_30) stats (balanced):")
    try:
        print(balanced_df['angular_z_next_30_sum_abs'].describe())
    except Exception as e:
        print(f"Could not display stats: {e}")
    
    # Show info about added future columns
    future_columns = ['manual_linear_x_next_10', 'manual_angular_z_next_10',
                     'manual_linear_x_next_30', 'manual_linear_x_next_60', 
                     'manual_angular_z_next_30', 'manual_angular_z_next_60']
    
    print(f"\nFuture prediction columns added to output:")
    for col in future_columns:
        if col in balanced_df_clean.columns:
            print(f"  {col}: Available (contains sequences of future values)")
            # Show a sample of the first sequence
            try:
                sample_seq = eval(balanced_df_clean[col].iloc[0])
                print(f"    Sample (first 5 values): {sample_seq[:5]}")
                print(f"    Sequence length: {len(sample_seq)}")
            except:
                print(f"    Sample: {balanced_df_clean[col].iloc[0][:50]}...")
    
    print(f"\nBalancing criteria:")
    print(f"  - Categories based on sum of abs(manual_angular_z_next_30)")
    print(f"  - Threshold: >= {args.threshold} for high_angular_activity")
    print(f"  - Threshold: < {args.threshold} for low_angular_activity")

if __name__ == "__main__":
    main()