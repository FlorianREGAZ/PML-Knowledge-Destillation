import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_ghostnetv3_width_accuracy():
    """
    Plot the best accuracy of GhostNetV3 small models over their width.
    X-axis: Width multiplier
    Y-axis: Best test accuracy
    """
    
    # Define the mapping from version to actual width multiplier
    version_to_width = {
        1: 1.0,
        2: 1.3,
        3: 1.6,
        4: 1.9,
        5: 2.2,
        6: 2.8,
        7: 3.4
    }
    
    # Read the CSV file with best accuracies
    csv_path = os.path.join('exported_data', 'best_accuracies_comparison.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter for GhostNetV3 models with default training only
    ghostnet_data = df[(df['model'] == 'GhostNetV3') & (df['experiment_type'] == 'default')]
    
    if ghostnet_data.empty:
        print("No GhostNetV3 default training data found in the CSV!")
        return
    
    # Add width column based on version
    ghostnet_data = ghostnet_data.copy()
    ghostnet_data['width'] = ghostnet_data['version'].map(version_to_width)
    
    # Sort by width for better plotting
    ghostnet_data = ghostnet_data.sort_values('width')
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the data
    plt.plot(ghostnet_data['width'], ghostnet_data['best_accuracy'], 
            'o-', linewidth=2, markersize=8, label='GhostNetV3-Small', color='blue')
    
    # Add value labels on points
    for _, row in ghostnet_data.iterrows():
        plt.annotate(f'{row["best_accuracy"]:.2f}%', 
                    (row['width'], row['best_accuracy']),
                    textcoords="offset points", xytext=(-8,25), ha='center',
                    fontsize=14, color='blue')
    
    # Customize the plot
    plt.xlabel('Width Multiplier', fontsize=16, fontweight='bold')
    plt.ylabel('Best Test Accuracy (%)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    
    # Set x-axis to show all widths
    all_widths = sorted(ghostnet_data['width'].unique())
    plt.xticks(all_widths)
    
    # Set y-axis range to show differences better
    min_acc = ghostnet_data['best_accuracy'].min()
    max_acc = ghostnet_data['best_accuracy'].max()
    margin = (max_acc - min_acc) * 0.1
    plt.ylim(min_acc - margin, max_acc + margin)
    
    # Add some styling
    plt.tight_layout()
    
    # Print summary statistics
    print("\nGhostNetV3 Width vs Accuracy Summary:")
    print("="*50)
    print(f"Width range: {min(all_widths)} - {max(all_widths)}")
    print(f"Accuracy range: {min_acc:.2f}% - {max_acc:.2f}%")
    best_row = ghostnet_data.loc[ghostnet_data['best_accuracy'].idxmax()]
    print(f"Best performing width: {best_row['width']} (Accuracy: {best_row['best_accuracy']:.2f}%)")
    print("\nDetailed Results:")
    for _, row in ghostnet_data.iterrows():
        print(f"Width {row['width']:3.1f}: {row['best_accuracy']:6.2f}%")
    
    # Save the plot
    output_path = os.path.join('plots', 'ghostnetv3_width_vs_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Change to the project directory if running from plots folder
    if os.path.basename(os.getcwd()) == 'plots':
        os.chdir('..')
    
    plot_ghostnetv3_width_accuracy()