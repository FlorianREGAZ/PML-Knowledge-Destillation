import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up the plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_process_data(file_path, label):
    """Load CSV data and extract test accuracy at the end of each epoch"""
    df = pd.read_csv(file_path)
    
    # Group by epoch and take the last test_accuracy value for each epoch
    # (since test accuracy is evaluated at the end of each epoch)
    epoch_data = df.groupby('epoch').last().reset_index()
    
    return {
        'epochs': epoch_data['epoch'].values,
        'test_accuracy': epoch_data['test_accuracy'].values,
        'label': label
    }

def create_comparison_plot():
    """Create a comparison plot of test accuracy over epochs"""
    
    # Define the data files and their labels
    datasets = [
        {
            'file': 'train_default_20250529_v6_small_detailed.csv',
            'label': 'Default Training'
        },
        {
            'file': 'train_kd_ghostnetv3_20250601_small_detailed.csv',
            'label': 'Knowledge Distillation (ResNet18)'
        },
        # {
        #     'file': 'train_ta_20250611_v1_small_detailed.csv',
        #     'label': 'Teacher Assistant'
        # },
        # {
        #     'file': 'train_ensemble_20250612_v1_small_detailed.csv',
        #     'label': 'Ensemble Training'
        # }
    ]
    
    # Base path for data files
    base_path = '/Users/florianzager/University/Praktisches Machine Learning/Projekt/exported_data/'
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Load and plot data for each dataset
    for dataset in datasets:
        file_path = os.path.join(base_path, dataset['file'])
        
        # Check if file exists
        if os.path.exists(file_path):
            data = load_and_process_data(file_path, dataset['label'])
            plt.plot(data['epochs'], data['test_accuracy'], 
                    marker='o', linewidth=2, markersize=4, 
                    label=data['label'], alpha=0.8)
        else:
            print(f"Warning: File {dataset['file']} not found")
    
    # Customize the plot
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Comparison of Test Accuracy Over Epochs\nAcross Different Training Techniques', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Customize legend
    plt.legend(frameon=True, fancybox=True, shadow=True, 
               loc='lower right', fontsize=10)
    
    # Set axis limits and ticks
    plt.xlim(left=1)
    plt.ylim(bottom=0)
    
    # Add minor ticks
    plt.minorticks_on()
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_path = '/Users/florianzager/University/Praktisches Machine Learning/Projekt/plots/test_accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Plot saved to: {output_path}")
    
    # Show the plot
    plt.show()

def print_summary_statistics():
    """Print summary statistics for each training method"""
    
    datasets = [
        {
            'file': 'train_default_20250528_v1_small_detailed.csv',
            'label': 'Default Training'
        },
        {
            'file': 'train_kd_resnet18_20250531_small_detailed.csv',
            'label': 'Knowledge Distillation (ResNet18)'
        },
        {
            'file': 'train_ta_20250611_v1_small_detailed.csv',
            'label': 'Teacher Assistant'
        },
        {
            'file': 'train_ensemble_20250612_v1_small_detailed.csv',
            'label': 'Ensemble Training'
        }
    ]
    
    base_path = '/Users/florianzager/University/Praktisches Machine Learning/Projekt/exported_data/'
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for dataset in datasets:
        file_path = os.path.join(base_path, dataset['file'])
        
        if os.path.exists(file_path):
            data = load_and_process_data(file_path, dataset['label'])
            max_accuracy = max(data['test_accuracy'])
            final_accuracy = data['test_accuracy'][-1]
            max_epoch = data['epochs'][data['test_accuracy'].argmax()]
            
            print(f"\n{dataset['label']}:")
            print(f"  • Maximum Test Accuracy: {max_accuracy:.2f}% (Epoch {max_epoch})")
            print(f"  • Final Test Accuracy: {final_accuracy:.2f}%")
            print(f"  • Total Epochs: {len(data['epochs'])}")

if __name__ == "__main__":
    # Create the comparison plot
    create_comparison_plot()
    
    # Print summary statistics
    print_summary_statistics()