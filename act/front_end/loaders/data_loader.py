"""
📦 Data Loading for Front-End Integration

Pure data loading without specifications - clean separation of concerns.
Handles CSV files, image directories, and anchor data for VNNLIB specifications.
"""

from __future__ import annotations
import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import csv

# Optional pandas import with fallback
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas not available - CSV loading will use basic csv module")


class DatasetLoader:
    """Pure data loading without specifications"""
    
    def __init__(self):
        """Initialize DatasetLoader"""
        pass
        
    def load_csv_data(self, csv_path: str, 
                     label_column: str = "label",
                     skip_columns: List[str] = None) -> List[Tuple[np.ndarray, int]]:
        """
        Load CSV data as pure (sample, label) pairs
        
        Args:
            csv_path: Path to CSV file
            label_column: Name of label column (default: "label")
            skip_columns: Columns to skip when loading features
            
        Returns:
            List of (features_array, label_int) tuples
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If label column not found or data format invalid
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        try:
            # Load CSV with pandas if available, otherwise use basic csv
            if HAS_PANDAS:
                df = pd.read_csv(csv_path)
                return self._load_csv_with_pandas(df, label_column, skip_columns, csv_path)
            else:
                return self._load_csv_basic(csv_path, label_column, skip_columns)
                
        except Exception as e:
            raise ValueError(f"Failed to load CSV data from {csv_path}: {e}")
            
    def _load_csv_with_pandas(self, df, label_column: str, skip_columns: List[str], csv_path: str) -> List[Tuple[np.ndarray, int]]:
        """Load CSV using pandas"""
        # Handle different CSV formats
        if label_column in df.columns:
            # Standard format with named label column
            label_col = label_column
        elif "label" in df.columns:
            # Default label column
            label_col = "label"
        elif len(df.columns) > 1:
            # Assume last column is label
            label_col = df.columns[-1]
            print(f"📋 Using last column '{label_col}' as labels")
        else:
            raise ValueError("No label column found in CSV")
            
        # Extract labels
        labels = df[label_col].astype(int).values
        
        # Extract features (all columns except label and skip_columns)
        feature_columns = [col for col in df.columns if col != label_col]
        if skip_columns:
            feature_columns = [col for col in feature_columns if col not in skip_columns]
            
        features = df[feature_columns].values.astype(np.float32)
        
        # Create data pairs
        data_pairs = []
        for i in range(len(df)):
            sample = features[i]
            label = labels[i]
            data_pairs.append((sample, label))
            
        print(f"📦 Loaded {len(data_pairs)} samples from {csv_path}")
        print(f"📐 Feature shape: {features.shape[1:]}, Labels: {np.unique(labels)}")
        
        return data_pairs
        
    def _load_csv_basic(self, csv_path: str, label_column: str, skip_columns: List[str]) -> List[Tuple[np.ndarray, int]]:
        """Load CSV using basic csv module (fallback when pandas not available)"""
        data_pairs = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            # Find label column
            if label_column in headers:
                label_col = label_column
            elif "label" in headers:
                label_col = "label"
            else:
                # Use last column
                label_col = headers[-1]
                print(f"📋 Using last column '{label_col}' as labels")
                
            # Determine feature columns
            feature_columns = [col for col in headers if col != label_col]
            if skip_columns:
                feature_columns = [col for col in feature_columns if col not in skip_columns]
                
            # Read data
            for i, row in enumerate(reader):
                try:
                    features = np.array([float(row[col]) for col in feature_columns], dtype=np.float32)
                    label = int(row[label_col])
                    data_pairs.append((features, label))
                except (ValueError, KeyError) as e:
                    print(f"⚠️  Skipping row {i}: {e}")
                    
        print(f"📦 Loaded {len(data_pairs)} samples from {csv_path}")
        if data_pairs:
            print(f"📐 Feature shape: {data_pairs[0][0].shape}, Labels: {set(pair[1] for pair in data_pairs)}")
        
        return data_pairs
            
    def load_image_directory(self, image_dir: str, 
                           label_mapping: Dict[str, int] = None,
                           supported_extensions: List[str] = None) -> List[Tuple[Image.Image, int]]:
        """
        Load images from directory as (PIL.Image, label) pairs
        
        Args:
            image_dir: Path to directory containing images
            label_mapping: Optional mapping from subdirectory names to label integers
            supported_extensions: List of supported file extensions (default: ['.png', '.jpg', '.jpeg'])
            
        Returns:
            List of (PIL.Image, label_int) tuples
            
        Directory structure options:
            1. Flat directory: images/img1.png, img2.png (all label 0)
            2. Subdirectories: images/class0/img1.png, images/class1/img2.png
            3. With label_mapping: custom subdirectory name to label mapping
        """
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
        if supported_extensions is None:
            supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            
        image_dir_path = Path(image_dir)
        data_pairs = []
        
        # Check if directory has subdirectories (class-based organization)
        subdirs = [d for d in image_dir_path.iterdir() if d.is_dir()]
        
        if subdirs:
            # Subdirectory-based labels
            print(f"📁 Found {len(subdirs)} subdirectories, using as class labels")
            
            for subdir in sorted(subdirs):
                # Determine label for this subdirectory
                if label_mapping and subdir.name in label_mapping:
                    label = label_mapping[subdir.name]
                else:
                    # Use directory name as label (try to convert to int, otherwise use index)
                    try:
                        label = int(subdir.name)
                    except ValueError:
                        label = sorted([d.name for d in subdirs]).index(subdir.name)
                        
                # Load images from subdirectory
                subdir_images = self._load_images_from_path(subdir, supported_extensions)
                for img in subdir_images:
                    data_pairs.append((img, label))
                    
                print(f"📂 Loaded {len(subdir_images)} images from {subdir.name} (label: {label})")
                
        else:
            # Flat directory - all images get label 0
            print(f"📁 Flat directory structure, assigning label 0 to all images")
            images = self._load_images_from_path(image_dir_path, supported_extensions)
            for img in images:
                data_pairs.append((img, 0))
                
        print(f"📦 Total loaded: {len(data_pairs)} images")
        return data_pairs
        
    def _load_images_from_path(self, path: Path, 
                              supported_extensions: List[str]) -> List[Image.Image]:
        """Helper to load images from a single directory path"""
        images = []
        
        for file_path in path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    img = Image.open(file_path)
                    # Convert to RGB if needed (handles RGBA, grayscale, etc.)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"⚠️  Failed to load {file_path}: {e}")
                    
        return images
        
    def get_dataset_info(self, path: str) -> Dict[str, Any]:
        """
        Get information about a dataset without fully loading it
        
        Args:
            path: Path to dataset (CSV file or image directory)
            
        Returns:
            Dictionary with dataset information
        """
        info = {"path": path, "type": "unknown", "status": "unknown"}
        
        try:
            if os.path.isfile(path) and path.endswith('.csv'):
                # CSV file info
                df = pd.read_csv(path, nrows=5)  # Sample first few rows
                info.update({
                    "type": "csv",
                    "columns": list(df.columns),
                    "estimated_samples": sum(1 for _ in open(path)) - 1,  # Rough count
                    "feature_columns": len(df.columns) - 1,  # Assume last is label
                    "sample_dtypes": df.dtypes.to_dict(),
                    "status": "✅ CSV file accessible"
                })
                
            elif os.path.isdir(path):
                # Image directory info
                path_obj = Path(path)
                subdirs = [d for d in path_obj.iterdir() if d.is_dir()]
                
                # Count images
                image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
                total_images = 0
                
                if subdirs:
                    class_counts = {}
                    for subdir in subdirs:
                        count = len([f for f in subdir.iterdir() 
                                   if f.suffix.lower() in image_extensions])
                        class_counts[subdir.name] = count
                        total_images += count
                        
                    info.update({
                        "type": "image_directory",
                        "structure": "class_subdirectories",
                        "classes": list(class_counts.keys()),
                        "class_counts": class_counts,
                        "total_images": total_images,
                        "status": "✅ Image directory with classes"
                    })
                else:
                    total_images = len([f for f in path_obj.iterdir() 
                                     if f.suffix.lower() in image_extensions])
                    info.update({
                        "type": "image_directory", 
                        "structure": "flat",
                        "total_images": total_images,
                        "status": "✅ Flat image directory"
                    })
                    
            else:
                info["status"] = "❌ Path not found or unsupported format"
                
        except Exception as e:
            info["status"] = f"❌ Error accessing dataset: {e}"
            
        return info


def demo_data_loader():
    """Demo function to test DatasetLoader functionality"""
    print("📦 DatasetLoader Demo")
    print("=" * 50)
    
    loader = DatasetLoader()
    
    # Test CSV loading
    csv_path = "../data/MNIST_csv/mnist_first_100_samples.csv"
    if os.path.exists(csv_path):
        print(f"\n📊 Testing CSV loading: {csv_path}")
        try:
            data_pairs = loader.load_csv_data(csv_path)
            print(f"✅ Loaded {len(data_pairs)} CSV samples")
            if data_pairs:
                sample, label = data_pairs[0]
                print(f"📋 First sample: shape={sample.shape}, label={label}")
        except Exception as e:
            print(f"❌ CSV loading failed: {e}")
    else:
        print(f"⚠️  CSV file not found: {csv_path}")
        
    # Test dataset info
    print(f"\n📋 Testing dataset info:")
    if os.path.exists(csv_path):
        info = loader.get_dataset_info(csv_path)
        print(f"CSV info: {info['status']}")
        if 'estimated_samples' in info:
            print(f"  Samples: {info['estimated_samples']}, Features: {info['feature_columns']}")
            
    # Test image directory (if exists)
    image_dirs = ["../data/images", "../examples/images", "./test_images"]
    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            print(f"\n🖼️  Testing image directory: {img_dir}")
            try:
                data_pairs = loader.load_image_directory(img_dir)
                print(f"✅ Loaded {len(data_pairs)} images")
                if data_pairs:
                    img, label = data_pairs[0]
                    print(f"📋 First image: size={img.size}, mode={img.mode}, label={label}")
            except Exception as e:
                print(f"❌ Image loading failed: {e}")
            break
    else:
        print("\n⚠️  No image directories found for testing")


if __name__ == "__main__":
    demo_data_loader()