import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import glob
from tqdm import tqdm
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import seaborn as sns
from collections import defaultdict
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import gc

warnings.filterwarnings('ignore')

def load_pickle_safely(filepath):
    """Safely load pickle file"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None

class EffectiveWeightVisualization:
    def __init__(self, base_dir="incremental_weights"):
        self.base_dir = base_dir
        self.time_points = self._discover_time_points()
        
    def _discover_time_points(self):
        """Discover all time points"""
        pattern = os.path.join(self.base_dir, "epoch_*_iter_*")
        time_dirs = sorted(glob.glob(pattern))
        
        time_points = []
        for dir_path in time_dirs:
            if os.path.isdir(dir_path):
                dir_name = os.path.basename(dir_path)
                try:
                    parts = dir_name.split('_')
                    epoch = int(parts[1])
                    iteration = int(parts[3])
                    time_points.append((epoch, iteration, dir_path))
                except (ValueError, IndexError):
                    continue
        
        print(f"Found {len(time_points)} time points")
        return sorted(time_points)
    
    def load_time_point_data(self, epoch, iteration, max_samples=1000):
        """Load samples for the given time point with sample limit"""
        dir_path = os.path.join(self.base_dir, f"epoch_{epoch:02d}_iter_{iteration:03d}")
        
        print(f"Loading samples from: {dir_path}")
        
        if not os.path.exists(dir_path):
            print(f"Directory does not exist: {dir_path}")
            return None
        
        # Load sample files
        pattern = os.path.join(dir_path, "sample_*.pkl")
        files = sorted(glob.glob(pattern))
        
        print(f"Found {len(files)} sample files")
        
        if not files:
            print("No sample files found")
            return None
        
        # Limit samples to avoid memory issues
        if len(files) > max_samples:
            files = files[:max_samples]
            print(f"Limited to {max_samples} samples for memory efficiency")
        
        all_weights = []
        all_labels = []
        
        for filepath in tqdm(files, desc=f"Loading E{epoch}I{iteration}"):
            data = load_pickle_safely(filepath)
            if data is not None and 'effective_weight' in data:
                all_weights.append(data['effective_weight'])
                all_labels.append(data['label'])
        
        print(f"Loaded {len(all_weights)} samples")
        
        if len(all_weights) < 10:
            print(f"Not enough samples: {len(all_weights)}")
            return None
        
        weights_array = np.array(all_weights)
        weights_flat = weights_array.reshape(weights_array.shape[0], -1)
        
        print(f"Data shape - weights_flat: {weights_flat.shape}")
        
        return {
            'weights_flat': weights_flat,
            'weights_array': weights_array,
            'labels': np.array(all_labels),
            'epoch': epoch,
            'iteration': iteration,
            'total_samples': len(all_weights),
            'n_classes': len(np.unique(all_labels))
        }
    
    def compute_dimensionality_reductions(self, weights_flat, labels, method='pca'):
        """Compute dimensionality reduction"""
        n_samples = len(weights_flat)
        
        print(f"Computing {method.upper()} for {n_samples} samples...")
        
        try:
            if method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
                result = reducer.fit_transform(weights_flat)
                print(f"PCA explained variance: {reducer.explained_variance_ratio_}")
                
            elif method == 'umap':
                n_neighbors = min(30, max(15, n_samples // 50))
                min_dist = 0.1
                
                print(f"UMAP parameters: n_neighbors={n_neighbors}")
                
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=42,
                    metric='euclidean',
                    low_memory=True,
                    n_epochs=200
                )
                result = reducer.fit_transform(weights_flat)
                
            elif method == 'tsne':
                perplexity = min(50, max(30, n_samples // 100))
                
                print(f"t-SNE parameters: perplexity={perplexity}")
                
                reducer = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=42,
                    n_iter=500,
                    verbose=0,
                    method='barnes_hut'
                )
                result = reducer.fit_transform(weights_flat)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            print(f"{method.upper()} completed successfully")
            return result, reducer
        
        except Exception as e:
            print(f"Error in {method}: {e}")
            return None, None

    def create_individual_animations(self, all_time_data):
        """Create separate animations for PCA, UMAP, and t-SNE and save individual frames"""
        print("Creating individual animations for each method...")
        
        # Filter out time points without valid data
        valid_time_data = []
        for data in all_time_data:
            if (data.get('reductions') and data.get('labels') is not None and 
                len(data['labels']) > 0):
                valid_time_data.append(data)
        
        if not valid_time_data:
            print("No valid data for animations")
            return
        
        print(f"Creating animations with {len(valid_time_data)} valid time points")
        
        methods = ['pca', 'umap', 'tsne']
        
        for method in methods:
            print(f"\n=== Creating {method.upper()} animation ===")
            self._create_single_method_animation(valid_time_data, method)

    def _create_single_method_animation(self, animation_data, method):
        """Create animation for a single dimensionality reduction method and save individual frames"""
        print(f"Creating {method} animation with {len(animation_data)} frames")
        
        # Create output directories
        save_dir = os.path.join(self.base_dir, "animations")
        frames_dir = os.path.join(save_dir, f"{method}_frames")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Pre-calculate global axis limits for consistency
        all_projections = []
        for data in animation_data:
            proj = data.get('reductions', {}).get(method)
            if proj is not None and len(proj) > 0:
                all_projections.append(proj)
        
        if all_projections:
            all_points = np.vstack(all_projections)
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
        else:
            x_min, x_max, y_min, y_max = -1, 1, -1, 1
            x_margin = y_margin = 0.2
        
        # Create figure for animation
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Set up the figure ONCE
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Create colorbar
        scatter = ax.scatter([], [], alpha=0.6, s=8, cmap='tab10')
        cbar = plt.colorbar(scatter, ax=ax, label='Class Label')
        scatter.remove()
        
        # Title
        title = ax.set_title('', fontweight='bold', fontsize=14)
        
        # First, save individual frames
        print("Saving individual frames...")
        frame_paths = []
        for frame_idx, data in enumerate(animation_data):
            epoch = data['epoch']
            iteration = data['iteration']
            reductions = data.get('reductions', {})
            labels = data.get('labels', [])
            
            # Clear previous scatter plot
            for collection in ax.collections[:]:
                collection.remove()
            
            # Create scatter plot for this frame
            if reductions.get(method) is not None and len(labels) > 0:
                projection = reductions[method]
                
                scatter = ax.scatter(projection[:, 0], projection[:, 1],
                                   c=labels, cmap='tab10', alpha=0.6, s=8)
                
                # Update colorbar
                cbar.update_normal(scatter)
                
                # Update title
                title.set_text(f'{method.upper()} Projection - Epoch {epoch}, Iteration {iteration}\n'
                             f'Samples: {data.get("total_samples", 0):,}')
                
                # Save individual frame
                frame_filename = f"{method}_epoch{epoch:02d}_iter{iteration:03d}.png"
                frame_path = os.path.join(frames_dir, frame_filename)
                plt.savefig(frame_path, dpi=120, bbox_inches='tight')
                frame_paths.append(frame_path)
                
                print(f"  Saved frame: {frame_filename}")
            
            # Clean up for next frame
            scatter.remove()
        
        print(f"Saved {len(frame_paths)} individual frames to {frames_dir}")
        
        # Now create animation
        def init():
            """Initialize animation - set empty state"""
            for collection in ax.collections[:]:
                collection.remove()
            title.set_text('')
            return []
        
        def animate(frame):
            """Update animation for each frame"""
            if frame >= len(animation_data):
                return []
                
            data = animation_data[frame]
            epoch = data['epoch']
            iteration = data['iteration']
            reductions = data.get('reductions', {})
            labels = data.get('labels', [])
            
            # Clear previous scatter plot
            for collection in ax.collections[:]:
                collection.remove()
            
            # Create new scatter plot
            if reductions.get(method) is not None and len(labels) > 0:
                projection = reductions[method]
                
                scatter = ax.scatter(projection[:, 0], projection[:, 1],
                                   c=labels, cmap='tab10', alpha=0.6, s=8)
                
                # Update colorbar
                cbar.update_normal(scatter)
                
                # Update title
                title.set_text(f'{method.upper()} Projection - Epoch {epoch}, Iteration {iteration}\n'
                             f'Samples: {data.get("total_samples", 0):,}')
            else:
                title.set_text(f'{method.upper()} - Epoch {epoch}, Iteration {iteration}\nData Not Available')
            
            return []
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(animation_data),
                           init_func=init, interval=1500, blit=False, repeat=True)
        
        # Save animation
        save_path = os.path.join(save_dir, f"{method}_evolution.gif")
        
        try:
            writer = PillowWriter(fps=1, bitrate=1000)
            anim.save(save_path, writer=writer, dpi=120)
            print(f"✓ Saved {method.upper()} animation: {save_path}")
        except Exception as e:
            print(f"✗ Failed to save {method.upper()} animation: {e}")
        
        plt.close(fig)
        gc.collect()

    def run_effective_weight_analysis(self):
        """Run effective weight visualization analysis"""
        print("Starting Effective Weight Visualization Analysis")
        print("=" * 60)
        print("Features:")
        print("• PCA, UMAP, and t-SNE dimensionality reduction") 
        print("• Individual animations for each method")
        print("• Individual frames saved as PNG files")
        print("• No cluster metrics analysis")
        print("=" * 60)
        
        print(f"Total time points found: {len(self.time_points)}")
        
        if not self.time_points:
            print("No time points found! Check if incremental_weights directory exists.")
            return []
        
        all_analysis_results = []
        
        # Process each time point
        for epoch, iteration, dir_path in self.time_points:
            try:
                print(f"\n{'='*50}")
                print(f"Processing Epoch {epoch}, Iteration {iteration}")
                print(f"{'='*50}")
                
                # Load samples for this time point
                time_data = self.load_time_point_data(epoch, iteration)
                
                if time_data is not None:
                    # Compute dimensionality reductions
                    methods = ['pca', 'umap', 'tsne']
                    reductions = {}
                    
                    for method in methods:
                        result, reducer = self.compute_dimensionality_reductions(
                            time_data['weights_flat'], time_data['labels'], method
                        )
                        reductions[method] = result
                    
                    # Store results
                    result = {
                        'epoch': epoch,
                        'iteration': iteration,
                        'reductions': reductions,
                        'labels': time_data['labels'],
                        'total_samples': time_data['total_samples'],
                        'n_classes': time_data['n_classes']
                    }
                    
                    all_analysis_results.append(result)
                    print(f"✓ Successfully processed Epoch {epoch}, Iteration {iteration}")
                    
                else:
                    print(f"✗ Failed to load data for Epoch {epoch}, Iteration {iteration}")
                    
                # Clean memory
                gc.collect()
                    
            except Exception as e:
                print(f"✗ Error processing Epoch {epoch}, Iteration {iteration}: {e}")
                continue
        
        # Create animations and individual frames
        if all_analysis_results:
            print(f"\n{'='*50}")
            print(f"Successfully processed {len(all_analysis_results)} time points")
            print("Creating animations and individual frames...")
            print(f"{'='*50}")
            
            # Create individual animations for each method
            self.create_individual_animations(all_analysis_results)
            
            # Print summary
            self.print_analysis_summary(all_analysis_results)
        
        else:
            print("No valid data found for analysis")
        
        return all_analysis_results

    def print_analysis_summary(self, all_analysis_results):
        """Print summary of analysis results"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        if not all_analysis_results:
            print("No results to summarize")
            return
        
        # Count epochs
        epochs = set()
        for result in all_analysis_results:
            epochs.add(result['epoch'])
        
        print(f"Total Time Points Analyzed: {len(all_analysis_results)}")
        print(f"Epochs Covered: {sorted(epochs)}")
        print(f"Average Samples per Time Point: {np.mean([r['total_samples'] for r in all_analysis_results]):.0f}")
        
        # Check which methods have valid data
        methods = ['pca', 'umap', 'tsne']
        method_stats = {}
        for method in methods:
            valid_count = sum(1 for r in all_analysis_results 
                            if r['reductions'].get(method) is not None)
            method_stats[method] = valid_count
        
        print(f"\nValid projections by method:")
        for method, count in method_stats.items():
            print(f"  {method.upper()}: {count}/{len(all_analysis_results)}")
        
        animations_dir = os.path.join(self.base_dir, "animations")
        print(f"\nGenerated files in: {animations_dir}")
        print("For each method (pca, umap, tsne):")
        print("  • [method]_evolution.gif - Animation file")
        print("  • [method]_frames/ - Directory with individual PNG frames")
        print("="*60)

def main():
    """Main function to run effective weight visualization"""
    print("EFFECTIVE WEIGHT VISUALIZATION")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = EffectiveWeightVisualization(base_dir="incremental_weights")
    
    # Run analysis
    results = analyzer.run_effective_weight_analysis()
    
    if results:
        print("\n✓ Analysis completed successfully!")
        print("\nGenerated files:")
        animations_dir = os.path.join("incremental_weights", "animations")
        print(f"Location: {animations_dir}")
        print("\nFor PCA, UMAP, and t-SNE:")
        print("  • [method]_evolution.gif - Complete animation")
        print("  • [method]_frames/ - All individual frames as PNG")
        print("\nExample frame names:")
        print("  • pca_epoch01_iter000.png")
        print("  • umap_epoch02_iter234.png")
        print("  • tsne_epoch02_iter936.png")
    else:
        print("\n✗ Analysis failed! Check the error messages above.")

if __name__ == "__main__":
    main()
