"""
📦 Streamlined Data Loading for Front-End Integration

Clean torch tensor data loading with dynamic discovery.
Focuses on essential methods for CSV loading and project resource discovery.
"""

from __future__ import annotations
import os
import torch
import csv
from typing import List, Tuple, Dict, Any
from pathlib import Path
import pandas as pd
from act.util.device_manager import get_current_settings


class DatasetLoader:
    """Pure torch tensor data loading leveraging global device settings"""
    
    def __init__(self):
        """Initialize DatasetLoader with global device settings"""
        self.device, self.dtype = get_current_settings()
            
    def _load_csv_pandas_torch(self, df, label_column: str, skip_columns: List[str], csv_path: str) -> List[Tuple[torch.Tensor, int]]:
        """Load CSV using pandas with torch tensors"""
        # Handle different CSV formats
        if label_column in df.columns:
            label_col = label_column
        elif "label" in df.columns:
            label_col = "label"
        elif len(df.columns) > 1:
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
            
        features = df[feature_columns].values.astype('float32')
        
        # Create torch tensor data pairs using global device settings
        data_pairs = []
        for i in range(len(df)):
            sample_tensor = torch.tensor(features[i], dtype=self.dtype, device=self.device)
            label = int(labels[i])
            data_pairs.append((sample_tensor, label))
            
        print(f"📦 Loaded {len(data_pairs)} samples from {csv_path}")
        print(f"📐 Feature shape: {features.shape[1:]}, Labels: {torch.unique(torch.tensor(labels))}")
        print(f"🔧 Using device: {self.device}, dtype: {self.dtype}")
        
        return data_pairs
    
    def load_csv_torch(self, csv_path: str, label_column: str = "label", skip_columns: List[str] = None) -> List[Tuple[torch.Tensor, int]]:
        """Load CSV file and return torch tensor data pairs"""
        if skip_columns is None:
            skip_columns = []
            
        df = pd.read_csv(csv_path)
        return self._load_csv_pandas_torch(df, label_column, skip_columns, csv_path)
    
    def load_mnist_csv(self, csv_path: str = None) -> List[Tuple[torch.Tensor, int]]:
        """Convenience method for MNIST CSV loading with project defaults"""
        if csv_path is None:
            csv_path = "data/MNIST_csv/mnist_first_100_samples.csv"
        return self.load_csv_torch(csv_path, label_column="label")
    
    def discover_all_datasets(self) -> Dict[str, List[str]]:
        """Comprehensively discover all datasets in the project"""
        datasets = {
            "csv": [],
            "json": [],
            "raw": [],
            "vnnlib": [],
            "other": []
        }
        
        # Search entire data directory tree
        data_root = Path("data")
        if data_root.exists():
            for file_path in data_root.rglob("*"):
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    if suffix == ".csv":
                        datasets["csv"].append(str(file_path))
                    elif suffix == ".json":
                        datasets["json"].append(str(file_path))
                    elif suffix == ".vnnlib":
                        datasets["vnnlib"].append(str(file_path))
                    elif suffix in [".txt", ".dat", ".bin"]:
                        datasets["other"].append(str(file_path))
                        
                elif file_path.is_dir():
                    # Include data directories that might contain samples
                    if any(child.suffix.lower() in [".csv", ".json", ".png", ".jpg"] 
                           for child in file_path.rglob("*") if child.is_file()):
                        datasets["raw"].append(str(file_path))
                    
        return datasets
    
    def load_json_spec(self, json_path: str) -> Dict[str, Any]:
        """Load JSON specification file"""
        import json
        with open(json_path, 'r') as f:
            spec_data = json.load(f)
        return spec_data
    
    def load_for_act_backend(self, csv_path: str) -> Dict[str, torch.Tensor]:
        """Load data in format suitable for ACT backend"""
        data_list = self.load_csv_torch(csv_path)
        
        if not data_list:
            return {
                "features": torch.empty(0, device=self.device, dtype=self.dtype),
                "labels": torch.empty(0, device=self.device, dtype=torch.long),
                "num_samples": 0,
                "feature_dim": 0
            }
        
        # Separate features and labels into tensors
        features = torch.stack([item[0] for item in data_list])
        labels = torch.tensor([item[1] for item in data_list], device=self.device)
        
        return {
            "features": features,
            "labels": labels,
            "num_samples": len(data_list),
            "feature_dim": features.shape[1] if len(features.shape) > 1 else features.shape[0]
        }
    
    def load_all_for_act_backend(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load all discovered datasets for ACT backend"""
        discovered = self.discover_all_datasets()
        act_ready_data = {}
        
        for csv_path in discovered["csv"]:
            try:
                dataset_name = Path(csv_path).stem
                act_ready_data[dataset_name] = self.load_for_act_backend(csv_path)
                print(f"✅ Prepared '{dataset_name}' for ACT backend")
            except Exception as e:
                print(f"❌ Failed to prepare {csv_path}: {e}")
                
        return act_ready_data
