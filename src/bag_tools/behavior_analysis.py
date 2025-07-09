#!/usr/bin/env python3

# FIRST: Fix numpy compatibility before any other imports
import numpy as np

# Compatibility fix for numpy/pandas version issues
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

# Set matplotlib backend for headless/SSH environment
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
import sys
import warnings
import math
import json
warnings.filterwarnings('ignore')

# Manual implementation of statistical functions (no scipy needed)
def manual_ttest_rel(a, b):
    """Simple paired t-test implementation"""
    diff = np.array(a) - np.array(b)
    n = len(diff)
    if n < 2:
        return 0, 0.5
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return float('inf') if mean_diff != 0 else 0, 0.001 if mean_diff != 0 else 1.0
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    # Simplified p-value approximation
    p_value = 0.001 if abs(t_stat) > 3 else (0.05 if abs(t_stat) > 2 else 0.5)
    return t_stat, p_value

# Manual implementation of sklearn functions
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

print("🚗 Behavior Analysis Tool (Scipy-Free Version)")
print("✅ Using manual statistical calculations")

def load_synchronized_csv(csv_path):
    """Load synchronized dataset from CSV"""
    print("📊 Loading synchronized CSV dataset...")
    
    try:
        # Load CSV and clean column names (remove asterisks)
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.replace('*', '', regex=False)
        
        print(f"✅ Loaded {len(df)} synchronized samples")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Convert timestamps to relative time in seconds
        df['time_sec'] = (df['image_timestamp'] - df['image_timestamp'].min()) / 1e9
        
        # Calculate steering errors
        df['angular_error'] = df['auto_angular_z'] - df['manual_angular_z']
        df['linear_error'] = df['auto_linear_x'] - df['manual_linear_x']
        df['absolute_angular_error'] = np.abs(df['angular_error'])
        
        # Categorize steering commands
        steering_threshold = 0.1
        df['manual_direction'] = np.where(df['manual_angular_z'] > steering_threshold, 'left',
                                         np.where(df['manual_angular_z'] < -steering_threshold, 'right', 'straight'))
        df['auto_direction'] = np.where(df['auto_angular_z'] > steering_threshold, 'left',
                                       np.where(df['auto_angular_z'] < -steering_threshold, 'right', 'straight'))
        df['direction_match'] = df['manual_direction'] == df['auto_direction']
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def calculate_comprehensive_statistics(df):
    """Calculate detailed statistics for synchronized data"""
    print("\n📈 Comprehensive Statistical Analysis")
    print("=" * 60)
    
    # Basic statistics
    manual_stats = {
        'count': len(df),
        'angular_z_mean': df['manual_angular_z'].mean(),
        'angular_z_std': df['manual_angular_z'].std(),
        'angular_z_min': df['manual_angular_z'].min(),
        'angular_z_max': df['manual_angular_z'].max(),
        'linear_x_mean': df['manual_linear_x'].mean(),
        'turns_left': (df['manual_angular_z'] > 0.1).sum(),
        'turns_right': (df['manual_angular_z'] < -0.1).sum(),
        'straight': (abs(df['manual_angular_z']) <= 0.1).sum(),
    }
    
    auto_stats = {
        'count': len(df),
        'angular_z_mean': df['auto_angular_z'].mean(),
        'angular_z_std': df['auto_angular_z'].std(),
        'angular_z_min': df['auto_angular_z'].min(),
        'angular_z_max': df['auto_angular_z'].max(),
        'linear_x_mean': df['auto_linear_x'].mean(),
        'turns_left': (df['auto_angular_z'] > 0.1).sum(),
        'turns_right': (df['auto_angular_z'] < -0.1).sum(),
        'straight': (abs(df['auto_angular_z']) <= 0.1).sum(),
    }
    
    print("🎮 MANUAL DRIVING:")
    print(f"   Samples: {manual_stats['count']}")
    print(f"   Speed (linear_x): {manual_stats['linear_x_mean']:.3f} m/s")
    print(f"   Steering (angular_z): {manual_stats['angular_z_mean']:.3f} ± {manual_stats['angular_z_std']:.3f} rad/s")
    print(f"   Range: [{manual_stats['angular_z_min']:.3f}, {manual_stats['angular_z_max']:.3f}]")
    print(f"   Commands: {manual_stats['turns_left']} left, {manual_stats['turns_right']} right, {manual_stats['straight']} straight")
    
    print("\n🤖 AUTONOMOUS DRIVING:")
    print(f"   Samples: {auto_stats['count']}")
    print(f"   Speed (linear_x): {auto_stats['linear_x_mean']:.3f} m/s")
    print(f"   Steering (angular_z): {auto_stats['angular_z_mean']:.3f} ± {auto_stats['angular_z_std']:.3f} rad/s")
    print(f"   Range: [{auto_stats['angular_z_min']:.3f}, {auto_stats['angular_z_max']:.3f}]")
    print(f"   Commands: {auto_stats['turns_left']} left, {auto_stats['turns_right']} right, {auto_stats['straight']} straight")
    
    # Error analysis
    mse = mean_squared_error(df['manual_angular_z'], df['auto_angular_z'])
    mae = mean_absolute_error(df['manual_angular_z'], df['auto_angular_z'])
    rmse = np.sqrt(mse)
    
    print("\n❌ ERROR ANALYSIS:")
    print(f"   Mean Absolute Error (MAE): {mae:.4f} rad/s")
    print(f"   Root Mean Square Error (RMSE): {rmse:.4f} rad/s")
    print(f"   Mean error: {df['angular_error'].mean():.4f} rad/s")
    print(f"   Error std: {df['angular_error'].std():.4f} rad/s")
    print(f"   Max absolute error: {df['absolute_angular_error'].max():.4f} rad/s")
    
    # Direction accuracy
    direction_accuracy = df['direction_match'].mean() * 100
    print(f"   Direction accuracy: {direction_accuracy:.1f}%")
    
    # Correlation
    correlation = df['manual_angular_z'].corr(df['auto_angular_z'])
    print(f"   Steering correlation: {correlation:.4f}")
    
    # Statistical significance using manual t-test
    t_stat, p_value = manual_ttest_rel(df['manual_angular_z'], df['auto_angular_z'])
    print(f"   Paired t-test p-value: {p_value:.6f}")
    
    # Synchronization quality if available
    print("\n⏰ SYNCHRONIZATION QUALITY:")
    if 'auto_time_diff' in df.columns:
        print(f"   Auto sync time diff: {df['auto_time_diff'].mean():.3f} ± {df['auto_time_diff'].std():.3f} s")
    if 'manual_time_diff' in df.columns:
        print(f"   Manual sync time diff: {df['manual_time_diff'].mean():.3f} ± {df['manual_time_diff'].std():.3f} s")
    if 'joy_time_diff' in df.columns:
        print(f"   Joy sync time diff: {df['joy_time_diff'].mean():.3f} ± {df['joy_time_diff'].std():.3f} s")
    
    return manual_stats, auto_stats, {'mae': mae, 'rmse': rmse, 'correlation': correlation, 'direction_accuracy': direction_accuracy}

def create_synchronized_visualizations(df, output_dir):
    """Create comprehensive visualizations for synchronized data"""
    print("\n🎨 Creating synchronized visualizations...")
    
    plt.style.use('default')
    # Set a nice color cycle manually
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Time series comparison with both manual and autonomous
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(df['time_sec'], df['manual_angular_z'], 'b-', alpha=0.8, label='Manual', linewidth=1.5)
    ax1.plot(df['time_sec'], df['auto_angular_z'], 'r-', alpha=0.8, label='Autonomous', linewidth=1.5)
    ax1.fill_between(df['time_sec'], df['manual_angular_z'], df['auto_angular_z'], 
                     alpha=0.3, color='gray', label='Difference')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angular Z (rad/s)')
    ax1.set_title('🕒 Synchronized Steering Commands')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error over time
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(df['time_sec'], df['angular_error'], 'g-', alpha=0.7, linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Error (Auto - Manual)')
    ax2.set_title('❌ Steering Error Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot: Manual vs Autonomous
    ax3 = plt.subplot(4, 3, 3)
    scatter = ax3.scatter(df['manual_angular_z'], df['auto_angular_z'], alpha=0.6, s=20, c=df['time_sec'], cmap='viridis')
    plt.colorbar(scatter, ax=ax3, label='Time (s)')
    # Perfect prediction line
    min_val = min(df['manual_angular_z'].min(), df['auto_angular_z'].min())
    max_val = max(df['manual_angular_z'].max(), df['auto_angular_z'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect prediction')
    ax3.set_xlabel('Manual Angular Z (rad/s)')
    ax3.set_ylabel('Autonomous Angular Z (rad/s)')
    ax3.set_title('🎯 Manual vs Autonomous Steering')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4 = plt.subplot(4, 3, 4)
    ax4.hist(df['angular_error'], bins=50, alpha=0.7, color='orange', density=True)
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax4.axvline(x=df['angular_error'].mean(), color='red', linestyle='-', alpha=0.8, label=f'Mean: {df["angular_error"].mean():.3f}')
    ax4.set_xlabel('Error (Auto - Manual)')
    ax4.set_ylabel('Density')
    ax4.set_title('📊 Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Absolute error over time
    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(df['time_sec'], df['absolute_angular_error'], 'orange', alpha=0.7, linewidth=1)
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Absolute Error (rad/s)')
    ax5.set_title('📏 Absolute Error Over Time')
    ax5.grid(True, alpha=0.3)
    
    # 6. Speed comparison
    ax6 = plt.subplot(4, 3, 6)
    ax6.plot(df['time_sec'], df['manual_linear_x'], 'b-', alpha=0.8, label='Manual', linewidth=1.5)
    ax6.plot(df['time_sec'], df['auto_linear_x'], 'r-', alpha=0.8, label='Autonomous', linewidth=1.5)
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Linear X (m/s)')
    ax6.set_title('🏎️ Speed Commands')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Direction accuracy heatmap (manual implementation)
    ax7 = plt.subplot(4, 3, 7)
    try:
        direction_confusion = pd.crosstab(df['manual_direction'], df['auto_direction'], normalize='index') * 100
        
        # Create heatmap manually
        im = ax7.imshow(direction_confusion.values, cmap='Blues', aspect='auto')
        ax7.set_xticks(range(len(direction_confusion.columns)))
        ax7.set_yticks(range(len(direction_confusion.index)))
        ax7.set_xticklabels(direction_confusion.columns)
        ax7.set_yticklabels(direction_confusion.index)
        
        # Add text annotations
        for i in range(len(direction_confusion.index)):
            for j in range(len(direction_confusion.columns)):
                value = direction_confusion.iloc[i, j]
                ax7.text(j, i, f'{value:.1f}', ha='center', va='center', 
                        color='white' if value > 50 else 'black')
        
        plt.colorbar(im, ax=ax7)
        ax7.set_title('🎯 Direction Prediction Accuracy (%)')
        ax7.set_xlabel('Autonomous Direction')
        ax7.set_ylabel('Manual Direction')
    except Exception as e:
        ax7.text(0.5, 0.5, f'Direction analysis\nerror: {str(e)[:30]}...', 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('🎯 Direction Analysis (Error)')
    
    # 8. Error vs steering magnitude
    ax8 = plt.subplot(4, 3, 8)
    scatter8 = ax8.scatter(np.abs(df['manual_angular_z']), df['absolute_angular_error'], 
                alpha=0.6, s=15, c=df['time_sec'], cmap='plasma')
    plt.colorbar(scatter8, ax=ax8, label='Time (s)')
    ax8.set_xlabel('|Manual Angular Z| (rad/s)')
    ax8.set_ylabel('Absolute Error (rad/s)')
    ax8.set_title('🔍 Error vs Steering Magnitude')
    ax8.grid(True, alpha=0.3)
    
    # 9. Joystick analysis
    ax9 = plt.subplot(4, 3, 9)
    if 'joy_axis_0' in df.columns and 'joy_axis_1' in df.columns:
        scatter9 = ax9.scatter(df['joy_axis_0'], df['joy_axis_1'], alpha=0.6, s=15, 
                   c=df['manual_angular_z'], cmap='RdBu_r')
        plt.colorbar(scatter9, ax=ax9, label='Manual Angular Z')
        ax9.set_xlabel('Joy Axis 0 (L/R)')
        ax9.set_ylabel('Joy Axis 1 (F/B)')
        ax9.set_title('🎮 Joystick Input vs Steering')
    else:
        ax9.text(0.5, 0.5, 'Joystick data\nnot available', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('🎮 Joystick Analysis (N/A)')
    ax9.grid(True, alpha=0.3)
    
    # 10. Rolling error statistics
    ax10 = plt.subplot(4, 3, 10)
    window_size = max(10, len(df) // 50)  # Adaptive window size
    rolling_mae = df['absolute_angular_error'].rolling(window=window_size, center=True).mean()
    rolling_std = df['angular_error'].rolling(window=window_size, center=True).std()
    
    ax10.plot(df['time_sec'], rolling_mae, 'r-', label=f'Rolling MAE (window={window_size})', linewidth=2)
    ax10.plot(df['time_sec'], rolling_std, 'b-', label=f'Rolling Std (window={window_size})', linewidth=2)
    ax10.set_xlabel('Time (seconds)')
    ax10.set_ylabel('Error Metrics')
    ax10.set_title('📈 Rolling Error Statistics')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Synchronization quality
    ax11 = plt.subplot(4, 3, 11)
    sync_plotted = False
    if 'auto_time_diff' in df.columns:
        ax11.plot(df['time_sec'], df['auto_time_diff'], 'r-', alpha=0.7, label='Auto sync', linewidth=1)
        sync_plotted = True
    if 'manual_time_diff' in df.columns:
        ax11.plot(df['time_sec'], df['manual_time_diff'], 'b-', alpha=0.7, label='Manual sync', linewidth=1)
        sync_plotted = True
    if 'joy_time_diff' in df.columns:
        ax11.plot(df['time_sec'], df['joy_time_diff'], 'g-', alpha=0.7, label='Joy sync', linewidth=1)
        sync_plotted = True
    
    if sync_plotted:
        ax11.set_xlabel('Time (seconds)')
        ax11.set_ylabel('Sync Time Diff (s)')
        ax11.set_title('⏰ Synchronization Quality')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
    else:
        ax11.text(0.5, 0.5, 'Sync timing data\nnot available', ha='center', va='center', transform=ax11.transAxes)
        ax11.set_title('⏰ Synchronization Quality (N/A)')
    
    # 12. Summary metrics
    ax12 = plt.subplot(4, 3, 12)
    metrics = ['MAE', 'RMSE', 'Correlation', 'Direction Acc %']
    values = [
        df['absolute_angular_error'].mean(),
        np.sqrt(mean_squared_error(df['manual_angular_z'], df['auto_angular_z'])),
        df['manual_angular_z'].corr(df['auto_angular_z']),
        df['direction_match'].mean() * 100
    ]
    colors = ['red', 'orange', 'green', 'blue']
    bars = ax12.bar(metrics, values, color=colors, alpha=0.7)
    ax12.set_title('📊 Summary Metrics')
    ax12.set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax12.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / "synchronized_behavior_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Visualization saved: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)

def analyze_performance_patterns(df):
    """Analyze performance patterns in different scenarios"""
    print("\n🔍 Performance Pattern Analysis")
    print("=" * 60)
    
    # Performance by steering intensity
    df['steering_intensity'] = pd.cut(np.abs(df['manual_angular_z']), 
                                     bins=[0, 0.1, 0.5, 1.0, float('inf')], 
                                     labels=['Straight', 'Light', 'Medium', 'Sharp'])
    
    intensity_stats = df.groupby('steering_intensity')['absolute_angular_error'].agg(['count', 'mean', 'std'])
    print("📐 Performance by steering intensity:")
    for intensity in intensity_stats.index:
        if pd.notna(intensity):  # Skip NaN categories
            count = intensity_stats.loc[intensity, 'count']
            mean_err = intensity_stats.loc[intensity, 'mean']
            std_err = intensity_stats.loc[intensity, 'std']
            print(f"   {intensity:>8}: {count:>4} samples, MAE: {mean_err:.4f} ± {std_err:.4f}")
    
    # Performance by time (learning/adaptation)
    df['time_quartile'] = pd.qcut(df['time_sec'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    time_stats = df.groupby('time_quartile')['absolute_angular_error'].agg(['count', 'mean', 'std'])
    print("\n⏰ Performance over time:")
    for quartile in time_stats.index:
        count = time_stats.loc[quartile, 'count']
        mean_err = time_stats.loc[quartile, 'mean']
        std_err = time_stats.loc[quartile, 'std']
        print(f"   {quartile}: {count:>4} samples, MAE: {mean_err:.4f} ± {std_err:.4f}")
    
    # Check for improvement/degradation over time
    first_half_error = df.iloc[:len(df)//2]['absolute_angular_error'].mean()
    second_half_error = df.iloc[len(df)//2:]['absolute_angular_error'].mean()
    improvement = ((first_half_error - second_half_error) / first_half_error) * 100
    
    print(f"\n📈 Performance trend:")
    print(f"   First half MAE: {first_half_error:.4f}")
    print(f"   Second half MAE: {second_half_error:.4f}")
    if improvement > 0:
        print(f"   ✅ Improved by {improvement:.1f}%")
    else:
        print(f"   📉 Degraded by {-improvement:.1f}%")
    
    # Joystick correlation analysis
    if 'joy_axis_0' in df.columns and 'joy_axis_1' in df.columns:
        print(f"\n🎮 Joystick correlation analysis:")
        joy_corr_manual = df['joy_axis_0'].corr(df['manual_angular_z'])
        joy_corr_auto = df['joy_axis_0'].corr(df['auto_angular_z'])
        print(f"   Joy axis 0 vs Manual steering: {joy_corr_manual:.4f}")
        print(f"   Joy axis 0 vs Auto steering: {joy_corr_auto:.4f}")

def save_detailed_results(df, output_dir, stats):
    """Save detailed analysis results"""
    # Save processed dataframe
    df_path = output_dir / "analyzed_synchronized_data.csv"
    df.to_csv(df_path, index=False)
    print(f"💾 Analyzed data saved: {df_path}")
    
    # Save summary statistics
    summary_path = output_dir / "analysis_summary.json"
    summary = {
        'total_samples': len(df),
        'mae': stats['mae'],
        'rmse': stats['rmse'],
        'correlation': stats['correlation'],
        'direction_accuracy': stats['direction_accuracy'],
        'mean_error': df['angular_error'].mean(),
        'error_std': df['angular_error'].std(),
        'max_absolute_error': df['absolute_angular_error'].max(),
        'sync_quality': {},
        'manual_stats': {
            'mean_angular_z': df['manual_angular_z'].mean(),
            'std_angular_z': df['manual_angular_z'].std(),
            'mean_linear_x': df['manual_linear_x'].mean()
        },
        'auto_stats': {
            'mean_angular_z': df['auto_angular_z'].mean(),
            'std_angular_z': df['auto_angular_z'].std(),
            'mean_linear_x': df['auto_linear_x'].mean()
        }
    }
    
    # Add sync quality if available
    if 'auto_time_diff' in df.columns:
        summary['sync_quality']['auto_time_diff_mean'] = df['auto_time_diff'].mean()
    if 'manual_time_diff' in df.columns:
        summary['sync_quality']['manual_time_diff_mean'] = df['manual_time_diff'].mean()
    if 'joy_time_diff' in df.columns:
        summary['sync_quality']['joy_time_diff_mean'] = df['joy_time_diff'].mean()
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved: {summary_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 behavior_analysis.py <synchronized_csv>")
        print("Example: python3 behavior_analysis.py synchronized_dataset.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    output_dir = csv_path.parent
    
    print("=" * 60)
    print(f"📁 Synchronized CSV: {csv_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load synchronized data
    df = load_synchronized_csv(csv_path)
    if df is None:
        return
    
    # Calculate comprehensive statistics
    manual_stats, auto_stats, error_stats = calculate_comprehensive_statistics(df)
    
    # Analyze performance patterns
    analyze_performance_patterns(df)
    
    # Create visualizations
    create_synchronized_visualizations(df, output_dir)
    
    # Save detailed results
    save_detailed_results(df, output_dir, error_stats)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Key Performance Metrics:")
    print(f"   - Total synchronized samples: {len(df)}")
    print(f"   - Mean Absolute Error: {error_stats['mae']:.4f} rad/s")
    print(f"   - Steering Correlation: {error_stats['correlation']:.4f}")
    print(f"   - Direction Accuracy: {error_stats['direction_accuracy']:.1f}%")
    print(f"   - Results saved in: {output_dir}")

if __name__ == "__main__":
    main()