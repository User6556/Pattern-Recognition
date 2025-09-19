import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues
import matplotlib.pyplot as plt
from PIL import Image
import os

class HopfieldNetwork:
    """
    Hopfield Neural Network for Pattern Recognition
    
    This class implements a Hopfield network that can store binary patterns
    and recall them from noisy or incomplete inputs.
    """
    
    def __init__(self, size):
        """
        Initialize the Hopfield network.
        
        Args:
            size (int): Size of the input patterns (number of neurons)
        """
        self.size = size
        self.weights = np.zeros((size, size))
        self.patterns = []
    
    def train(self, patterns, pattern_names=None):
        """
        Train the network using Hebbian learning rule.
        
        Args:
            patterns (list): List of binary patterns to store
            pattern_names (list): Optional list of pattern names
        """
        self.patterns = patterns
        self.pattern_names = pattern_names if pattern_names else [f"Pattern_{i}" for i in range(len(patterns))]
        self.weights = np.zeros((self.size, self.size))
        
        # Apply Hebbian learning rule
        for pattern in patterns:
            # Ensure pattern is a numpy array
            pattern = np.array(pattern)
            # Update weights: W += outer product of pattern with itself
            self.weights += np.outer(pattern, pattern)
        
        # Remove self-connections (diagonal elements)
        np.fill_diagonal(self.weights, 0)
        
        # Normalize weights
        self.weights = self.weights / len(patterns)
    
    def recall(self, input_pattern, max_iterations=100, mode='synchronous'):
        """
        Recall a pattern from noisy input using synchronous or asynchronous updates.
        
        Args:
            input_pattern (array): Noisy input pattern
            max_iterations (int): Maximum number of iterations
            mode (str): 'synchronous' or 'asynchronous' update mode
            
        Returns:
            tuple: (recalled_pattern, converged, iterations)
        """
        # Ensure input is a numpy array
        state = np.array(input_pattern, dtype=float)
        
        for iteration in range(max_iterations):
            # Store previous state to check for convergence
            prev_state = state.copy()
            
            if mode == 'synchronous':
                # Synchronous update: update all neurons simultaneously
                net_inputs = np.dot(self.weights, state)
                state = np.where(net_inputs >= 0, 1, -1)
            else:
                # Asynchronous update: update neurons one by one
                for i in range(self.size):
                    # Compute weighted sum of inputs
                    net_input = np.dot(self.weights[i], state)
                    # Apply sign activation function
                    state[i] = 1 if net_input >= 0 else -1
            
            # Check for convergence
            if np.array_equal(state, prev_state):
                return state, True, iteration + 1
        
        # If not converged within max_iterations
        return state, False, max_iterations
    
    def energy(self, pattern):
        """
        Calculate the energy of a given pattern.
        
        Args:
            pattern (array): Input pattern
            
        Returns:
            float: Energy value
        """
        pattern = np.array(pattern)
        return -0.5 * np.dot(pattern, np.dot(self.weights, pattern))
    
    def pattern_similarity(self, pattern1, pattern2):
        """
        Calculate similarity between two patterns.
        
        Args:
            pattern1, pattern2 (array): Patterns to compare
            
        Returns:
            float: Similarity score (0 to 1)
        """
        pattern1 = np.array(pattern1)
        pattern2 = np.array(pattern2)
        
        # Calculate Hamming distance
        matches = np.sum(pattern1 == pattern2)
        return matches / len(pattern1)
    
    def hamming_distance(self, pattern1, pattern2):
        """
        Calculate Hamming distance between two patterns.
        
        Args:
            pattern1, pattern2 (array): Patterns to compare
            
        Returns:
            int: Number of differing bits
        """
        pattern1 = np.array(pattern1)
        pattern2 = np.array(pattern2)
        return np.sum(pattern1 != pattern2)
    
    def analyze_pattern_distances(self, input_pattern):
        """
        Analyze distances from input pattern to all stored patterns.
        
        Args:
            input_pattern (array): Input pattern to analyze
            
        Returns:
            dict: Dictionary with pattern names and their distances
        """
        distances = {}
        similarities = {}
        
        for i, stored_pattern in enumerate(self.patterns):
            # Use stored pattern names
            pattern_name = self.pattern_names[i] if i < len(self.pattern_names) else f"Pattern_{i}"
            
            hamming_dist = self.hamming_distance(input_pattern, stored_pattern)
            similarity = self.pattern_similarity(input_pattern, stored_pattern)
            
            distances[pattern_name] = hamming_dist
            similarities[pattern_name] = similarity
        
        return distances, similarities
    
    def recall_with_debug(self, input_pattern, max_iterations=100):
        """
        Enhanced recall function with debugging information.
        
        Args:
            input_pattern (array): Noisy input pattern
            max_iterations (int): Maximum number of iterations
            
        Returns:
            tuple: (recalled_pattern, converged, iterations, debug_info)
        """
        # Analyze initial distances
        initial_distances, initial_similarities = self.analyze_pattern_distances(input_pattern)
        
        # Perform recall
        recalled_pattern, converged, iterations = self.recall(input_pattern, max_iterations)
        
        # Analyze final distances
        final_distances, final_similarities = self.analyze_pattern_distances(recalled_pattern)
        
        # Find best match
        best_match = min(final_distances, key=final_distances.get)
        
        debug_info = {
            'initial_distances': initial_distances,
            'initial_similarities': initial_similarities,
            'final_distances': final_distances,
            'final_similarities': final_similarities,
            'best_match': best_match,
            'converged': converged,
            'iterations': iterations
        }
        
        return recalled_pattern, converged, iterations, debug_info

def save_preprocessed_debug(input_pattern, grid_size=(7, 7), save_path='static/preprocessed_debug.png'):
    """
    Save a visualization of the preprocessed input pattern for debugging.
    
    Args:
        input_pattern (array): The preprocessed binary pattern
        grid_size (tuple): Size of the pattern grid
        save_path (str): Path to save the debug image
    """
    try:
        # Create figure for the preprocessed pattern
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        fig.suptitle('Preprocessed Input (What Network Sees)', fontsize=14)
        
        # Convert pattern back to 2D for visualization
        pattern_2d = input_pattern.reshape(grid_size)
        
        # Plot the pattern
        ax.imshow(pattern_2d, cmap='RdYlBu', vmin=-1, vmax=1)
        ax.set_title('Binary Pattern')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid lines
        for i in range(grid_size[0] + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
        for j in range(grid_size[1] + 1):
            ax.axvline(j - 0.5, color='black', linewidth=0.5)
        
        # Add text annotations showing values
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                value = pattern_2d[i, j]
                color = 'white' if value == 1 else 'black'
                ax.text(j, i, f'{int(value)}', ha='center', va='center', 
                       color=color, fontweight='bold', fontsize=8)
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Save the plot
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error saving preprocessed debug image: {e}")
        return False

def preprocess_image(image_path, target_size=(7, 7)):
    """
    Preprocess an uploaded image for the Hopfield network.
    Enhanced preprocessing with better noise handling and adaptive thresholding.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (width, height)
        
    Returns:
        numpy.array: Binary pattern (-1 and 1)
    """
    try:
        # Open and convert to grayscale
        img = Image.open(image_path).convert("L")
        
        # Resize to target size using LANCZOS for better quality, then NEAREST for final step
        if img.size != target_size:
            # First resize with high quality
            intermediate_size = (target_size[0] * 4, target_size[1] * 4)
            img = img.resize(intermediate_size, Image.LANCZOS)
            # Then resize to final size with NEAREST to avoid anti-aliasing
            img = img.resize(target_size, Image.NEAREST)
        
        # Convert to numpy array
        arr = np.array(img)
        
        # Use adaptive thresholding for better results
        # Calculate optimal threshold using Otsu's method approximation
        hist, _ = np.histogram(arr, bins=256, range=(0, 256))
        total_pixels = arr.size
        
        # Find optimal threshold
        best_threshold = 128  # default
        max_variance = 0
        
        for threshold in range(1, 255):
            # Background weight and mean
            w_bg = np.sum(hist[:threshold]) / total_pixels
            if w_bg == 0:
                continue
            mean_bg = np.sum(np.arange(threshold) * hist[:threshold]) / np.sum(hist[:threshold])
            
            # Foreground weight and mean  
            w_fg = np.sum(hist[threshold:]) / total_pixels
            if w_fg == 0:
                continue
            mean_fg = np.sum(np.arange(threshold, 256) * hist[threshold:]) / np.sum(hist[threshold:])
            
            # Between-class variance
            variance = w_bg * w_fg * (mean_bg - mean_fg) ** 2
            
            if variance > max_variance:
                max_variance = variance
                best_threshold = threshold
        
        # Apply threshold to create bipolar vector
        # Use the adaptive threshold, but ensure it's reasonable
        if best_threshold < 50 or best_threshold > 200:
            best_threshold = 128  # fallback to standard threshold
            
        arr = np.where(arr < best_threshold, 1, -1)
        
        # Optional: Apply morphological operations to clean up the pattern
        # This helps with noisy images
        kernel_size = max(1, min(target_size) // 7)  # adaptive kernel size
        if kernel_size > 1:
            # Simple erosion-dilation to clean noise
            from scipy.ndimage import binary_erosion, binary_dilation
            binary_img = (arr == 1)
            binary_img = binary_erosion(binary_img, iterations=1)
            binary_img = binary_dilation(binary_img, iterations=1)
            arr = np.where(binary_img, 1, -1)
        
        # Flatten to 1D vector
        return arr.flatten()
        
    except ImportError:
        # Fallback if scipy is not available
        print("Note: scipy not available, using basic preprocessing")
        # Use simpler preprocessing
        img = Image.open(image_path).convert("L")
        img = img.resize(target_size, Image.NEAREST)
        arr = np.array(img)
        
        # Use mean as threshold for better results than fixed 128
        threshold = np.mean(arr)
        arr = np.where(arr < threshold, 1, -1)
        return arr.flatten()
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def create_training_patterns():
    """
    Create hardcoded binary patterns for letters P and Q (7x7 grid).
    Larger patterns provide better distinctiveness and reduced misclassification.
    
    Returns:
        dict: Dictionary containing pattern names and their binary representations
    """
    patterns = {}
    
    # Letter P (7x7 grid) - More distinctive pattern
    # 1 1 1 1 1 1 0
    # 1 0 0 0 0 0 1
    # 1 0 0 0 0 0 1
    # 1 1 1 1 1 1 0
    # 1 0 0 0 0 0 0
    # 1 0 0 0 0 0 0
    # 1 0 0 0 0 0 0
    pattern_P = np.array([
        [ 1,  1,  1,  1,  1,  1, -1],
        [ 1, -1, -1, -1, -1, -1,  1],
        [ 1, -1, -1, -1, -1, -1,  1],
        [ 1,  1,  1,  1,  1,  1, -1],
        [ 1, -1, -1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1, -1, -1]
    ]).flatten()
    
    # Letter Q (7x7 grid) - More distinctive with clear tail
    # 0 1 1 1 1 1 0
    # 1 0 0 0 0 0 1
    # 1 0 0 0 0 0 1
    # 1 0 0 0 0 0 1
    # 1 0 0 0 1 0 1
    # 1 0 0 0 0 1 1
    # 0 1 1 1 1 1 1
    pattern_Q = np.array([
        [-1,  1,  1,  1,  1,  1, -1],
        [ 1, -1, -1, -1, -1, -1,  1],
        [ 1, -1, -1, -1, -1, -1,  1],
        [ 1, -1, -1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1, -1,  1],
        [ 1, -1, -1, -1, -1,  1,  1],
        [-1,  1,  1,  1,  1,  1,  1]
    ]).flatten()
    
    # Letter R (7x7 grid) - Additional pattern for better training
    # 1 1 1 1 1 1 0
    # 1 0 0 0 0 0 1
    # 1 0 0 0 0 0 1
    # 1 1 1 1 1 1 0
    # 1 0 0 1 0 0 0
    # 1 0 0 0 1 0 0
    # 1 0 0 0 0 1 0
    pattern_R = np.array([
        [ 1,  1,  1,  1,  1,  1, -1],
        [ 1, -1, -1, -1, -1, -1,  1],
        [ 1, -1, -1, -1, -1, -1,  1],
        [ 1,  1,  1,  1,  1,  1, -1],
        [ 1, -1, -1,  1, -1, -1, -1],
        [ 1, -1, -1, -1,  1, -1, -1],
        [ 1, -1, -1, -1, -1,  1, -1]
    ]).flatten()
    
    patterns['P'] = pattern_P
    patterns['Q'] = pattern_Q
    patterns['R'] = pattern_R
    
    return patterns

def visualize_patterns(input_pattern, recalled_pattern, stored_patterns, 
                      save_path='static/result.png', grid_size=(7, 7)):
    """
    Create a visualization showing input, recalled, and stored patterns.
    
    Args:
        input_pattern (array): Original noisy input
        recalled_pattern (array): Pattern recalled by Hopfield network
        stored_patterns (dict): Dictionary of stored patterns
        save_path (str): Path to save the visualization
        grid_size (tuple): Size of the pattern grid
    """
    # Determine which stored pattern is most similar to recalled pattern
    best_match = None
    best_similarity = 0
    
    for name, pattern in stored_patterns.items():
        similarity = np.sum(recalled_pattern == pattern) / len(pattern)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Hopfield Network Pattern Recall', fontsize=16)
    
    # Convert patterns back to 2D for visualization
    input_2d = input_pattern.reshape(grid_size)
    recalled_2d = recalled_pattern.reshape(grid_size)
    stored_2d = stored_patterns[best_match].reshape(grid_size) if best_match else recalled_2d
    
    # Plot input pattern
    axes[0].imshow(input_2d, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[0].set_title('Noisy Input')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Plot recalled pattern
    axes[1].imshow(recalled_2d, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[1].set_title('Recalled Pattern')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # Plot stored pattern
    axes[2].imshow(stored_2d, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[2].set_title(f'Stored Pattern: {best_match}\n(Similarity: {best_similarity:.2f})')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    # Add grid lines for better visualization
    for ax in axes:
        for i in range(grid_size[0] + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
        for j in range(grid_size[1] + 1):
            ax.axvline(j - 0.5, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_match, best_similarity

def add_noise_to_pattern(pattern, noise_level=0.1):
    """
    Add noise to a pattern for testing purposes.
    
    Args:
        pattern (array): Original pattern
        noise_level (float): Fraction of bits to flip (0.0 to 1.0)
        
    Returns:
        numpy.array: Noisy pattern
    """
    noisy_pattern = pattern.copy()
    num_bits_to_flip = int(len(pattern) * noise_level)
    
    # Randomly select bits to flip
    indices_to_flip = np.random.choice(len(pattern), num_bits_to_flip, replace=False)
    
    # Flip the selected bits
    for idx in indices_to_flip:
        noisy_pattern[idx] = -noisy_pattern[idx]
    
    return noisy_pattern

if __name__ == "__main__":
    # Test the Hopfield network
    print("Testing Hopfield Network...")
    
    # Create training patterns
    patterns = create_training_patterns()
    print(f"Created {len(patterns)} training patterns: {list(patterns.keys())}")
    
    # Initialize and train the network
    network = HopfieldNetwork(size=49)  # 7x7 = 49 neurons
    pattern_names = list(patterns.keys())
    network.train(list(patterns.values()), pattern_names)
    print("Network trained successfully!")
    
    # Test with noisy patterns
    for pattern_name, clean_pattern in patterns.items():
        print(f"\nTesting pattern: {pattern_name}")
        
        # Add noise
        noisy_pattern = add_noise_to_pattern(clean_pattern, noise_level=0.2)
        
        # Calculate initial similarity
        initial_similarity = network.pattern_similarity(noisy_pattern, clean_pattern)
        print(f"Initial similarity: {initial_similarity:.2f}")
        
        # Recall pattern
        recalled, converged, iterations = network.recall(noisy_pattern)
        
        # Calculate final similarity
        final_similarity = network.pattern_similarity(recalled, clean_pattern)
        print(f"Recalled in {iterations} iterations (converged: {converged})")
        print(f"Final similarity: {final_similarity:.2f}")
        
        # Create visualization for 7x7 grid
        visualize_patterns(noisy_pattern, recalled, patterns, 
                         save_path=f'test_{pattern_name.lower()}_result.png', grid_size=(7, 7))
        print(f"Visualization saved as test_{pattern_name.lower()}_result.png")
