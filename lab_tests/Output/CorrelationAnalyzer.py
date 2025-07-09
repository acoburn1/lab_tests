import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class CorrelationAnalyzer:
    """
    A class for analyzing and visualizing correlation data from neural network training.
    Handles finding peak correlations, plotting matrices, and graphing correlation trends.
    """
    
    def __init__(self, npz_filepath):
        """
        Initialize the analyzer by loading data from an .npz file.
        
        Args:
            npz_filepath (str): Path to the .npz file containing training results
        """
        self.data = np.load(npz_filepath, allow_pickle=True)
        self.losses = self.data['losses']
        self.m_output_corrs = self.data['m_output_corrs']
        self.l_output_corrs = self.data['l_output_corrs']
        self.m_hidden_corrs = self.data['m_hidden_corrs']
        self.l_hidden_corrs = self.data['l_hidden_corrs']
        self.output_matrices = self.data['output_matrices']
        self.hidden_matrices = self.data['hidden_matrices']
        
    def find_highest_correlations(self):
        """
        Find the epochs with highest correlation values for each correlation type.
        
        Returns:
            dict: Dictionary containing highest correlation values and their epochs
        """
        results = {
            'modular_output': {
                'max_corr': np.max(self.m_output_corrs),
                'epoch': np.argmax(self.m_output_corrs),
                'matrix': self.output_matrices[np.argmax(self.m_output_corrs)]
            },
            'lattice_output': {
                'max_corr': np.max(self.l_output_corrs),
                'epoch': np.argmax(self.l_output_corrs),
                'matrix': self.output_matrices[np.argmax(self.l_output_corrs)]
            },
            'modular_hidden': {
                'max_corr': np.max(self.m_hidden_corrs),
                'epoch': np.argmax(self.m_hidden_corrs),
                'matrix': self.hidden_matrices[np.argmax(self.m_hidden_corrs)]
            },
            'lattice_hidden': {
                'max_corr': np.max(self.l_hidden_corrs),
                'epoch': np.argmax(self.l_hidden_corrs),
                'matrix': self.hidden_matrices[np.argmax(self.l_hidden_corrs)]
            }
        }
        
        return results
    
    def plot_peak_matrices(self, save_dir="Results/Plots", show_plots=True):
        """
        Plot matrices from epochs with highest correlations.
        
        Args:
            save_dir (str): Directory to save plots
            show_plots (bool): Whether to display plots interactively
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        peak_results = self.find_highest_correlations()
        
        # Create a 2x2 subplot for all peak matrices
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Matrices from Epochs with Highest Correlations', fontsize=16)
        
        # Modular Output Matrix
        ax = axes[0, 0]
        matrix = peak_results['modular_output']['matrix']
        epoch = peak_results['modular_output']['epoch']
        corr = peak_results['modular_output']['max_corr']
        
        sns.heatmap(matrix, ax=ax, annot=False, cmap="viridis", cbar=True, square=True)
        ax.set_title(f'Modular Output (Epoch {epoch+1}, r={corr:.3f})')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        
        # Lattice Output Matrix
        ax = axes[0, 1]
        matrix = peak_results['lattice_output']['matrix']
        epoch = peak_results['lattice_output']['epoch']
        corr = peak_results['lattice_output']['max_corr']
        
        sns.heatmap(matrix, ax=ax, annot=False, cmap="viridis", cbar=True, square=True)
        ax.set_title(f'Lattice Output (Epoch {epoch+1}, r={corr:.3f})')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        
        # Modular Hidden Matrix
        ax = axes[1, 0]
        matrix = peak_results['modular_hidden']['matrix']
        epoch = peak_results['modular_hidden']['epoch']
        corr = peak_results['modular_hidden']['max_corr']
        
        sns.heatmap(matrix, ax=ax, annot=False, cmap="viridis", cbar=True, square=True)
        ax.set_title(f'Modular Hidden (Epoch {epoch+1}, r={corr:.3f})')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        
        # Lattice Hidden Matrix
        ax = axes[1, 1]
        matrix = peak_results['lattice_hidden']['matrix']
        epoch = peak_results['lattice_hidden']['epoch']
        corr = peak_results['lattice_hidden']['max_corr']
        
        sns.heatmap(matrix, ax=ax, annot=False, cmap="viridis", cbar=True, square=True)
        ax.set_title(f'Lattice Hidden (Epoch {epoch+1}, r={corr:.3f})')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{save_dir}/peak_correlation_matrices.png", dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
    def plot_correlation_trends(self, save_dir="Results/Plots", show_plots=True):
        """
        Plot correlation trends over epochs with epochs on x-axis and loss as a 5th line on secondary y-axis.
        
        Args:
            save_dir (str): Directory to save plots
            show_plots (bool): Whether to display plots interactively
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        epochs = range(1, len(self.losses) + 1)
        
        # Create the plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Plot correlation lines on primary y-axis
        line1 = ax1.plot(epochs, self.m_output_corrs, 'b-', linewidth=2, label='Modular Output', marker='o', markersize=4)
        line2 = ax1.plot(epochs, self.l_output_corrs, 'r-', linewidth=2, label='Lattice Output', marker='s', markersize=4)
        line3 = ax1.plot(epochs, self.m_hidden_corrs, 'g--', linewidth=2, label='Modular Hidden', marker='^', markersize=4)
        line4 = ax1.plot(epochs, self.l_hidden_corrs, 'm--', linewidth=2, label='Lattice Hidden', marker='v', markersize=4)
        
        # Set up primary y-axis (correlations)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Pearson Correlation', fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        
        # Create secondary y-axis for loss
        ax2 = ax1.twinx()
        line5 = ax2.plot(epochs, self.losses, 'k-', linewidth=2, label='Loss', marker='d', markersize=3)
        ax2.set_ylabel('Loss', fontsize=12, color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Set x-axis to show whole number epochs
        ax1.set_xticks(epochs)
        ax1.set_xlim(1, len(epochs))
        
        # Combine legends from both axes
        lines = line1 + line2 + line3 + line4 + line5
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
        
        plt.title('Correlation Trends and Loss Over Training Epochs', fontsize=14)
        
        # Add annotations for peak correlation values
        peak_results = self.find_highest_correlations()
        for key, result in peak_results.items():
            epoch = result['epoch'] + 1  # Convert to 1-based indexing
            max_corr = result['max_corr']
            
            ax1.annotate(f'Max: {max_corr:.3f}', 
                        xy=(epoch, max_corr), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{save_dir}/correlation_trends_with_loss.png", dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def print_summary(self):
        """
        Print a summary of the highest correlation values and their epochs.
        """
        peak_results = self.find_highest_correlations()
        
        print("=== Correlation Analysis Summary ===\n")
        
        for key, result in peak_results.items():
            corr_type = key.replace('_', ' ').title()
            epoch = result['epoch'] + 1  # Convert to 1-based indexing
            max_corr = result['max_corr']
            loss_at_peak = self.losses[result['epoch']]
            
            print(f"{corr_type}:")
            print(f"  Highest Correlation: {max_corr:.4f}")
            print(f"  Occurred at Epoch: {epoch}")
            print(f"  Loss at Peak: {loss_at_peak:.6f}")
            print()
            
    def generate_full_analysis(self, save_dir="Results/Analysis", show_plots=True):
        """
        Generate a complete analysis including all plots and summary.
        
        Args:
            save_dir (str): Directory to save all outputs
            show_plots (bool): Whether to display plots interactively
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating correlation analysis...")
        
        # Print summary
        self.print_summary()
        
        # Generate all plots
        plot_dir = os.path.join(save_dir, "plots")
        self.plot_peak_matrices(save_dir=plot_dir, show_plots=show_plots)
        self.plot_correlation_trends(save_dir=plot_dir, show_plots=show_plots)  # This now shows epochs with loss on secondary y-axis
        
        # Save summary to text file
        summary_file = os.path.join(save_dir, "correlation_summary.txt")
        with open(summary_file, 'w') as f:
            peak_results = self.find_highest_correlations()
            f.write("=== Correlation Analysis Summary ===\n\n")
            
            for key, result in peak_results.items():
                corr_type = key.replace('_', ' ').title()
                epoch = result['epoch'] + 1
                max_corr = result['max_corr']
                loss_at_peak = self.losses[result['epoch']]
                
                f.write(f"{corr_type}:\n")
                f.write(f"  Highest Correlation: {max_corr:.4f}\n")
                f.write(f"  Occurred at Epoch: {epoch}\n")
                f.write(f"  Loss at Peak: {loss_at_peak:.6f}\n\n")
        
        print(f"Analysis complete! Results saved to: {save_dir}")