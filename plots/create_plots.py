import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from pathlib import Path

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_train_default_data():
    """Load all train_default CSV files and combine them into a single DataFrame"""
    
    # Get the directory path
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "exported_data"
    
    # Find all train_default files
    train_default_files = glob.glob(str(data_dir / "train_default_*_detailed.csv"))
    
    all_data = []
    
    for file_path in train_default_files:
        # Extract experiment name from filename
        filename = os.path.basename(file_path)
        # Extract version info (e.g., "v1", "v2", etc.)
        parts = filename.split('_')
        date_part = parts[2]  # e.g., "20250528"
        version_part = parts[3]  # e.g., "v1"
        experiment_name = f"{date_part}_{version_part}"
        
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Get test accuracy at the end of each epoch (last training_step per epoch)
        epoch_data = df.groupby('epoch').last().reset_index()
        epoch_data['experiment'] = experiment_name
        epoch_data['file_path'] = filename
        
        all_data.append(epoch_data[['epoch', 'test_accuracy', 'experiment', 'file_path']])
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def create_test_accuracy_evolution_plot():
    """Create a beautiful plot showing test accuracy evolution over epochs"""
    
    # Load data
    df = load_train_default_data()
    
    # Create the figure with a larger size for better readability
    plt.figure(figsize=(14, 8))
    
    # Create the line plot
    ax = sns.lineplot(
        data=df,
        x='epoch',
        y='test_accuracy',
        hue='experiment',
        marker='o',
        markersize=6,
        linewidth=2.5,
        alpha=0.8
    )
    
    # Customize the plot
    plt.title('Test Accuracy Evolution Over Epochs\n(Train Default Experiments)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    
    # Improve grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Customize legend
    plt.legend(title='Experiment', title_fontsize=12, fontsize=10, 
               frameon=True, fancybox=True, shadow=True, 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set y-axis to start from 0 and add some padding
    max_accuracy = df['test_accuracy'].max()
    plt.ylim(0, max_accuracy * 1.1)
    
    # Add subtle styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Improve tick formatting
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # Add a subtle background color
    ax.set_facecolor('#fafafa')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(__file__).parent / 'test_accuracy_evolution_train_default.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Plot saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    return df

def create_summary_statistics():
    """Create a summary table of the experiments"""
    
    df = load_train_default_data()
    
    # Calculate summary statistics
    summary = df.groupby('experiment').agg({
        'test_accuracy': ['max', 'mean', 'std', 'count'],
        'epoch': 'max'
    }).round(2)
    
    # Flatten column names
    summary.columns = ['Max_Accuracy', 'Mean_Accuracy', 'Std_Accuracy', 'Num_Epochs', 'Total_Epochs']
    summary = summary.drop('Total_Epochs', axis=1)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS - TRAIN DEFAULT EXPERIMENTS")
    print("="*60)
    print(summary.to_string())
    print("="*60)
    
    return summary

def create_final_accuracy_comparison():
    """Create a bar plot comparing final test accuracies"""
    
    df = load_train_default_data()
    
    # Get final accuracy for each experiment
    final_accuracies = df.groupby('experiment')['test_accuracy'].max().reset_index()
    final_accuracies = final_accuracies.sort_values('test_accuracy', ascending=False)
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(final_accuracies['experiment'], final_accuracies['test_accuracy'], 
                   color=sns.color_palette("viridis", len(final_accuracies)), 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, final_accuracies['test_accuracy']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Final Test Accuracy Comparison\n(Train Default Experiments)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Experiment', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set y-axis to start from 0
    plt.ylim(0, final_accuracies['test_accuracy'].max() * 1.1)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(__file__).parent / 'final_accuracy_comparison_train_default.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Plot saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Creating beautiful plots for train_default experiments...")
    print("Loading data...")
    
    # Create the main evolution plot
    df = create_test_accuracy_evolution_plot()
    
    # Create summary statistics
    summary = create_summary_statistics()
    
    # Create final accuracy comparison
    create_final_accuracy_comparison()
    
    print(f"\nAnalyzed {df['experiment'].nunique()} train_default experiments")
    print("All plots have been created and saved!")
