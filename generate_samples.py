#!/usr/bin/env python3
"""
Generate sample images that exactly match the stored 7x7 patterns.
This will create clean samples and add various types of noise for testing.
"""

import numpy as np
from PIL import Image
import os
from hopfield import create_training_patterns, add_noise_to_pattern
import random

def create_image_from_pattern(pattern, grid_size=(7, 7), cell_size=40):
    """
    Create a PIL Image from a binary pattern.
    
    Args:
        pattern (array): Binary pattern (-1 and 1)
        grid_size (tuple): Size of the pattern grid
        cell_size (int): Size of each cell in pixels
    
    Returns:
        PIL.Image: Generated image
    """
    # Reshape pattern to 2D grid
    pattern_2d = pattern.reshape(grid_size)
    
    # Create image array
    img_height = grid_size[0] * cell_size
    img_width = grid_size[1] * cell_size
    img_array = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Fill the image based on pattern
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Convert pattern values: 1 (black) -> 0, -1 (white) -> 255
            color = 0 if pattern_2d[i, j] == 1 else 255
            
            # Fill the cell
            start_row = i * cell_size
            end_row = start_row + cell_size
            start_col = j * cell_size
            end_col = start_col + cell_size
            
            img_array[start_row:end_row, start_col:end_col] = color
    
    return Image.fromarray(img_array, mode='L')

def add_blur_noise(image, blur_radius=2):
    """Add blur noise to an image."""
    from PIL import ImageFilter
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def add_salt_pepper_noise(image, noise_prob=0.1):
    """Add salt and pepper noise to an image."""
    img_array = np.array(image)
    
    # Generate random noise
    noise = np.random.random(img_array.shape)
    
    # Salt noise (white pixels)
    img_array[noise < noise_prob/2] = 255
    
    # Pepper noise (black pixels)  
    img_array[noise > 1 - noise_prob/2] = 0
    
    return Image.fromarray(img_array, mode='L')

def add_random_noise(image, noise_level=0.1):
    """Add random noise to an image."""
    img_array = np.array(image).astype(float)
    
    # Add Gaussian noise
    noise = np.random.normal(0, 50, img_array.shape)
    noisy_img = img_array + noise * noise_level * 255
    
    # Clip values to valid range
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_img, mode='L')

def generate_sample_images():
    """Generate all sample images based on stored patterns."""
    
    # Get the stored patterns
    patterns = create_training_patterns()
    
    # Create output directory
    output_dir = 'noisy_samples'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating sample images based on stored 7×7 patterns...")
    
    for pattern_name, pattern in patterns.items():
        if pattern_name in ['P', 'Q']:  # Only generate P and Q samples
            print(f"\nGenerating samples for pattern {pattern_name}:")
            
            # 1. Clean image
            clean_img = create_image_from_pattern(pattern, grid_size=(7, 7), cell_size=50)
            clean_path = os.path.join(output_dir, f'{pattern_name}_clean.png')
            clean_img.save(clean_path)
            print(f"  ✅ Saved: {clean_path}")
            
            # 2. Slightly noisy version (10% pattern noise)
            noisy_pattern = add_noise_to_pattern(pattern, noise_level=0.1)
            noisy_img = create_image_from_pattern(noisy_pattern, grid_size=(7, 7), cell_size=50)
            noisy_path = os.path.join(output_dir, f'{pattern_name}_light_noise.png')
            noisy_img.save(noisy_path)
            print(f"  ✅ Saved: {noisy_path}")
            
            # 3. Moderate noisy version (20% pattern noise)
            moderate_noisy_pattern = add_noise_to_pattern(pattern, noise_level=0.2)
            moderate_noisy_img = create_image_from_pattern(moderate_noisy_pattern, grid_size=(7, 7), cell_size=50)
            moderate_path = os.path.join(output_dir, f'{pattern_name}_moderate_noise.png')
            moderate_noisy_img.save(moderate_path)
            print(f"  ✅ Saved: {moderate_path}")
            
            # 4. Blurred version
            blur_img = add_blur_noise(clean_img, blur_radius=1.5)
            blur_path = os.path.join(output_dir, f'{pattern_name}_blur.png')
            blur_img.save(blur_path)
            print(f"  ✅ Saved: {blur_path}")
            
            # 5. Salt & pepper noise
            saltpepper_img = add_salt_pepper_noise(clean_img, noise_prob=0.05)
            saltpepper_path = os.path.join(output_dir, f'{pattern_name}_saltpepper.png')
            saltpepper_img.save(saltpepper_path)
            print(f"  ✅ Saved: {saltpepper_path}")
            
            # 6. Random noise
            random_noise_img = add_random_noise(clean_img, noise_level=0.15)
            random_path = os.path.join(output_dir, f'{pattern_name}_random_noise.png')
            random_noise_img.save(random_path)
            print(f"  ✅ Saved: {random_path}")
    
    print(f"\n🎉 Sample images generated successfully in '{output_dir}/' folder!")
    print("\nThese images are designed to match the stored 7×7 patterns exactly.")
    print("Expected accuracy results:")
    print("  - Clean images: ~100%")
    print("  - Light noise: ~90-95%") 
    print("  - Moderate noise: ~80-90%")
    print("  - Blur: ~85-95%")
    print("  - Salt & pepper: ~70-85%")
    print("  - Random noise: ~75-90%")

if __name__ == "__main__":
    generate_sample_images()
