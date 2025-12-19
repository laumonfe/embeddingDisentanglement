"""
Script to prepare test split data for visualization:
1. Filters dataframe to test split only
2. Copies test images to static/images directory (once per unique image)
3. Saves filtered dataframe to static directory
"""

import pandas as pd
import os
import shutil
from pathlib import Path

def prepare_test_data(csv_path, output_dir, base_image_dir=None):
    """
    Prepare test split data for visualization.
    
    Args:
        csv_path: Path to the metadata CSV file
        output_dir: Output directory for static files
        base_image_dir: Base directory where original images are stored (optional)
    """
    # Create output directories
    static_dir = Path(output_dir)
    images_dir = static_dir / "images"
    static_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataframe
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    test_df = df[df['split'] == 'test'].copy()
    print(f"Filtered to {len(test_df)} test samples (from {len(df)} total)")
    
    # Track copied images to avoid duplicates
    copied_images = {}  # original_path -> new_relative_path
    
    # Process images
    copied_count = 0
    missing_count = 0
    new_image_paths = []
    
    for idx, row in test_df.iterrows():
        if 'image_path' not in row or pd.isna(row['image_path']) or row['image_path'] == '':
            new_image_paths.append('')
            missing_count += 1
            continue
        
        original_path = row['image_path']
        
        # Check if we already copied this image
        if original_path in copied_images:
            new_image_paths.append(copied_images[original_path])
            continue
        
        # Convert to Path and handle both absolute and relative paths
        original_full_path = Path(original_path)
        
        # If path doesn't exist and base_image_dir is provided, try combining
        if not original_full_path.exists() and base_image_dir:
            # Strip any existing 'data/images' or similar prefix from original_path
            path_parts = Path(original_path).parts
            # Find the actual filename (last part)
            filename = path_parts[-1]
            original_full_path = Path(base_image_dir) / filename
        
        # Check if original image exists
        if not original_full_path.exists():
            # Try one more time: check if the path is already correct as-is
            alt_path = Path(original_path)
            if alt_path.exists():
                original_full_path = alt_path
            else:
                print(f"Warning: Image not found: {original_path}")
                new_image_paths.append('')
                copied_images[original_path] = ''
                missing_count += 1
                continue
        
        # Create new filename (use item_idx to avoid collisions)
        item_idx = row['item_idx']
        file_extension = original_full_path.suffix
        new_filename = f"{item_idx}{file_extension}"
        new_path = images_dir / new_filename
        
        # Copy image
        try:
            shutil.copy2(original_full_path, new_path)
            # Store relative path for web serving
            relative_path = f"visualization_explorer/static/images/{new_filename}"
            new_image_paths.append(relative_path)
            copied_images[original_path] = relative_path
            copied_count += 1
            if copied_count % 500 == 0:
                print(f"Copied {copied_count} images...")
        except Exception as e:
            print(f"Error copying {original_full_path}: {e}")
            new_image_paths.append('')
            copied_images[original_path] = ''
            missing_count += 1
    
    # Update dataframe with new image paths
    test_df['image_path'] = new_image_paths
    
    # Save filtered dataframe
    output_csv = static_dir / "test_metadata.csv"
    test_df.to_csv(output_csv, index=False)
    print(f"\nSaved test metadata to {output_csv}")
    print(f"Total test samples: {len(test_df)}")
    print(f"Unique images copied: {copied_count}")
    print(f"Images missing: {missing_count}")
    
    return test_df

if __name__ == "__main__":
    # Configuration
    csv_path = r"visualizations/visualization_explorer/static/test_metadata.csv"
    output_dir = r"visualizations/visualization_explorer/static"
    
    # Set to None since image_path already contains the full path
    base_image_dir = None
    
    # Run preparation
    test_df = prepare_test_data(csv_path, output_dir, base_image_dir)
    
    print("\nPreparation complete!")
    print(f"You can now use 'static/test_metadata.csv' in your visualization.")