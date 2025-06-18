import os
import re
import glob
import csv
from typing import Dict, List, Tuple, Optional

class TrainingLogAnalyzer:
    """
    Analyzer for training log files to extract and visualize training/test metrics.
    """
    
    def __init__(self, logs_directory: str = "interesting_logs"):
        self.logs_directory = logs_directory
        self.data = []
        self.parsed_logs = {}
        
    def parse_single_log(self, log_path: str) -> Dict:
        """
        Parse a single log file and extract training metrics.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            Dictionary containing parsed data
        """
        filename = os.path.basename(log_path)
        
        # Extract experiment info from filename
        experiment_info = self._parse_filename(filename)
        
        # Initialize data structures
        epochs = []
        test_losses = []
        test_accuracies = []
        training_steps = []
        training_losses = []
        training_accuracies = []
        best_accuracy = None
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                
                # Parse test results (after each epoch)
                test_match = re.search(r'Test Loss: ([\d.]+) \| Test Acc: ([\d.]+)%', line)
                if test_match:
                    test_loss = float(test_match.group(1))
                    test_acc = float(test_match.group(2))
                    test_losses.append(test_loss)
                    test_accuracies.append(test_acc)
                    epochs.append(len(test_losses))
                
                # Parse training progress (during epochs)
                train_match = re.search(r'Epoch (\d+) \| Step (\d+)/\d+ \| Loss: ([\d.]+) \| Acc: ([\d.]+)%', line)
                if train_match:
                    epoch = int(train_match.group(1))
                    step = int(train_match.group(2))
                    loss = float(train_match.group(3))
                    acc = float(train_match.group(4))
                    
                    training_steps.append((epoch, step))
                    training_losses.append(loss)
                    training_accuracies.append(acc)
                
                # Parse final best accuracy
                best_match = re.search(r'Training complete\. Best Test Accuracy: ([\d.]+)%', line)
                if best_match:
                    best_accuracy = float(best_match.group(1))
        
        except Exception as e:
            print(f"Error parsing {log_path}: {e}")
            return {}
        
        return {
            'filename': filename,
            'experiment_type': experiment_info['type'],
            'model': experiment_info['model'],
            'date': experiment_info['date'],
            'version': experiment_info['version'],
            'epochs': epochs,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies,
            'training_steps': training_steps,
            'training_losses': training_losses,
            'training_accuracies': training_accuracies,
            'best_accuracy': best_accuracy,
            'total_epochs': len(test_losses)
        }
    
    def _parse_filename(self, filename: str) -> Dict:
        """
        Extract experiment information from filename.
        
        Examples:
        - train_default_20250528_v1_small.log
        - train_kd_resnet18_20250531_small.log
        - train_ensemble_20250612_v1_small.log
        """
        # Remove .log extension
        name = filename.replace('.log', '')
        
        # Parse different patterns
        if 'train_default' in name:
            match = re.search(r'train_default_(\d+)(?:_v(\d+))?_small', name)
            if match:
                return {
                    'type': 'default',
                    'model': 'GhostNetV3',
                    'date': match.group(1),
                    'version': match.group(2) or '1'
                }
        
        elif 'train_kd_' in name:
            match = re.search(r'train_kd_([^_]+)_(\d+)(?:_v(\d+))?_small', name)
            if match:
                return {
                    'type': 'knowledge_distillation',
                    'model': match.group(1),
                    'date': match.group(2),
                    'version': match.group(3) or '1'
                }
        
        elif 'train_ensemble' in name:
            match = re.search(r'train_ensemble_(\d+)_v(\d+)_small', name)
            if match:
                return {
                    'type': 'ensemble',
                    'model': 'ensemble',
                    'date': match.group(1),
                    'version': match.group(2)
                }
        
        elif 'train_ta' in name:
            match = re.search(r'train_ta_(\d+)_v(\d+)_small', name)
            if match:
                return {
                    'type': 'teacher_assistant',
                    'model': 'teacher_assistant',
                    'date': match.group(1),
                    'version': match.group(2)
                }
        
        # Fallback
        return {
            'type': 'unknown',
            'model': 'unknown',
            'date': 'unknown',
            'version': '1'
        }
    
    def parse_all_logs(self) -> List[Dict]:
        """
        Parse all log files in the directory.
        
        Returns:
            List of parsed log data dictionaries
        """
        log_files = glob.glob(os.path.join(self.logs_directory, "*.log"))
        
        for log_file in log_files:
            print(f"Parsing {os.path.basename(log_file)}...")
            parsed_data = self.parse_single_log(log_file)
            if parsed_data:  # Only add if parsing was successful
                self.data.append(parsed_data)
                self.parsed_logs[parsed_data['filename']] = parsed_data
        
        print(f"Successfully parsed {len(self.data)} log files.")
        return self.data
    
    def get_summary_dict(self) -> List[Dict]:
        """
        Create a summary list with key metrics for each experiment.
        
        Returns:
            List of dictionaries with experiment summaries
        """
        if not self.data:
            self.parse_all_logs()
        
        summary_data = []
        for exp in self.data:
            summary_data.append({
                'filename': exp['filename'],
                'experiment_type': exp['experiment_type'],
                'model': exp['model'],
                'date': exp['date'],
                'version': exp['version'],
                'total_epochs': exp['total_epochs'],
                'best_accuracy': exp['best_accuracy'],
                'final_test_accuracy': exp['test_accuracies'][-1] if exp['test_accuracies'] else None,
                'final_test_loss': exp['test_losses'][-1] if exp['test_losses'] else None
            })
        
        return summary_data
    
    def get_experiment_data(self, experiment_name: str) -> Optional[Dict]:
        """
        Get data for a specific experiment by filename.
        
        Args:
            experiment_name: Name of the experiment (filename without .log)
            
        Returns:
            Experiment data dictionary or None if not found
        """
        if not self.data:
            self.parse_all_logs()
        
        for exp in self.data:
            if experiment_name in exp['filename']:
                return exp
        return None
    
    def get_experiments_by_type(self, exp_type: str) -> List[Dict]:
        """
        Get all experiments of a specific type.
        
        Args:
            exp_type: Type of experiment ('default', 'knowledge_distillation', etc.)
            
        Returns:
            List of experiment data dictionaries
        """
        if not self.data:
            self.parse_all_logs()
        
        return [exp for exp in self.data if exp['experiment_type'] == exp_type]
    
    def export_detailed_data_to_csv(self, output_dir: str = "exported_data"):
        """
        Export all parsed data to detailed CSV files including both training and test metrics.
        
        Args:
            output_dir: Directory to save CSV files
        """
        if not self.data:
            self.parse_all_logs()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export summary
        summary_data = self.get_summary_dict()
        summary_file = os.path.join(output_dir, "experiments_summary.csv")
        
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            if summary_data:
                writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
                writer.writeheader()
                writer.writerows(summary_data)
        
        # Export detailed data for each experiment
        for exp in self.data:
            # Create comprehensive detailed data
            detailed_data = []
            
            # Get maximum length for alignment
            max_epochs = len(exp['test_accuracies']) if exp['test_accuracies'] else 0
            
            # Organize training data by epoch
            training_by_epoch = {}
            for i, (epoch, step) in enumerate(exp['training_steps']):
                if epoch not in training_by_epoch:
                    training_by_epoch[epoch] = {'steps': [], 'losses': [], 'accuracies': []}
                training_by_epoch[epoch]['steps'].append(step)
                training_by_epoch[epoch]['losses'].append(exp['training_losses'][i])
                training_by_epoch[epoch]['accuracies'].append(exp['training_accuracies'][i])
            
            # Create rows for each epoch
            for epoch_idx in range(max_epochs):
                epoch_num = epoch_idx + 1
                row = {
                    'epoch': epoch_num,
                    'test_loss': exp['test_losses'][epoch_idx] if epoch_idx < len(exp['test_losses']) else None,
                    'test_accuracy': exp['test_accuracies'][epoch_idx] if epoch_idx < len(exp['test_accuracies']) else None,
                }
                
                # Add training data for this epoch
                if epoch_num in training_by_epoch:
                    train_data = training_by_epoch[epoch_num]
                    # Add data for each training step in this epoch
                    for j, step in enumerate(train_data['steps']):
                        step_row = row.copy()
                        step_row.update({
                            'training_step': step,
                            'training_loss': train_data['losses'][j],
                            'training_accuracy': train_data['accuracies'][j]
                        })
                        detailed_data.append(step_row)
                else:
                    # If no training data for this epoch, still add test data
                    row.update({
                        'training_step': None,
                        'training_loss': None,
                        'training_accuracy': None
                    })
                    detailed_data.append(row)
            
            # Save detailed data
            if detailed_data:
                filename = exp['filename'].replace('.log', '_detailed.csv')
                detailed_file = os.path.join(output_dir, filename)
                
                with open(detailed_file, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['epoch', 'training_step', 'training_loss', 'training_accuracy', 'test_loss', 'test_accuracy']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(detailed_data)
        
        # Export aggregated comparison data
        self._export_comparison_data(output_dir)
        
        print(f"Detailed data exported to {output_dir}/")
    
    def _export_comparison_data(self, output_dir: str):
        """
        Export comparison data across all experiments.
        
        Args:
            output_dir: Directory to save files
        """
        # Test accuracy comparison across epochs
        test_acc_comparison = []
        test_loss_comparison = []
        
        # Find maximum number of epochs across all experiments
        max_epochs = max(len(exp['test_accuracies']) for exp in self.data if exp['test_accuracies'])
        
        for epoch in range(1, max_epochs + 1):
            acc_row = {'epoch': epoch}
            loss_row = {'epoch': epoch}
            
            for exp in self.data:
                exp_name = f"{exp['experiment_type']}_{exp['model']}_v{exp['version']}"
                
                if epoch <= len(exp['test_accuracies']):
                    acc_row[exp_name] = exp['test_accuracies'][epoch - 1]
                    loss_row[exp_name] = exp['test_losses'][epoch - 1]
                else:
                    acc_row[exp_name] = None
                    loss_row[exp_name] = None
            
            test_acc_comparison.append(acc_row)
            test_loss_comparison.append(loss_row)
        
        # Save comparison files
        if test_acc_comparison:
            acc_file = os.path.join(output_dir, "test_accuracy_comparison.csv")
            with open(acc_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=test_acc_comparison[0].keys())
                writer.writeheader()
                writer.writerows(test_acc_comparison)
        
        if test_loss_comparison:
            loss_file = os.path.join(output_dir, "test_loss_comparison.csv")
            with open(loss_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=test_loss_comparison[0].keys())
                writer.writeheader()
                writer.writerows(test_loss_comparison)
        
        # Export best accuracies comparison
        best_acc_data = []
        for exp in self.data:
            if exp['best_accuracy'] is not None:
                best_acc_data.append({
                    'experiment': f"{exp['experiment_type']}_{exp['model']}_v{exp['version']}",
                    'filename': exp['filename'],
                    'experiment_type': exp['experiment_type'],
                    'model': exp['model'],
                    'version': exp['version'],
                    'best_accuracy': exp['best_accuracy'],
                    'total_epochs': exp['total_epochs']
                })
        
        if best_acc_data:
            best_acc_file = os.path.join(output_dir, "best_accuracies_comparison.csv")
            with open(best_acc_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=best_acc_data[0].keys())
                writer.writeheader()
                writer.writerows(best_acc_data)


# Usage example and convenience functions
def analyze_logs(logs_dir: str = "interesting_logs"):
    """
    Convenience function to create analyzer and parse logs.
    
    Args:
        logs_dir: Directory containing log files
        
    Returns:
        TrainingLogAnalyzer instance with parsed data
    """
    analyzer = TrainingLogAnalyzer(logs_dir)
    analyzer.parse_all_logs()
    return analyzer

def quick_summary(logs_dir: str = "interesting_logs"):
    """
    Quick function to get summary of all experiments.
    
    Args:
        logs_dir: Directory containing log files
        
    Returns:
        List of dictionaries with experiment summaries
    """
    analyzer = analyze_logs(logs_dir)
    return analyzer.get_summary_dict()

def export_all_data(logs_dir: str = "interesting_logs", output_dir: str = "exported_data"):
    """
    Export all training data to detailed CSV files.
    
    Args:
        logs_dir: Directory containing log files
        output_dir: Directory to save CSV files
    """
    analyzer = analyze_logs(logs_dir)
    analyzer.export_detailed_data_to_csv(output_dir)

if __name__ == "__main__":
    # Example usage
    print("Training Log Analytics")
    print("=" * 50)
    
    # Create analyzer and parse logs
    analyzer = analyze_logs()
    
    # Show summary
    print("\nExperiment Summary:")
    summary = analyzer.get_summary_dict()
    
    # Print summary in a readable format
    for exp in summary:
        print(f"  {exp['filename']}: {exp['experiment_type']} - {exp['model']} v{exp['version']}")
        print(f"    Best accuracy: {exp['best_accuracy']:.2f}%" if exp['best_accuracy'] else "    Best accuracy: N/A")
        print(f"    Total epochs: {exp['total_epochs']}")
        print()
    
    # Show best performing experiments
    valid_experiments = [exp for exp in summary if exp['best_accuracy'] is not None]
    if valid_experiments:
        best_exp = max(valid_experiments, key=lambda x: x['best_accuracy'])
        print(f"Best performing experiment:")
        print(f"  {best_exp['filename']}: {best_exp['best_accuracy']:.2f}%")
    
    # Export detailed data
    print("\nExporting detailed data to CSV...")
    analyzer.export_detailed_data_to_csv()
    
    print("\nAnalysis complete!")
