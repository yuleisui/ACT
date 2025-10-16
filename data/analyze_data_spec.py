#!/usr/bin/env python3
"""
ACT Framework Comprehensive Data & Specification Analysis Tool
Unified script for analyzing datasets, specifications, and providing practical examples.
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

# Check for optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

class ComprehensiveAnalyzer:
    """Unified analyzer for ACT framework data and specifications."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the analyzer."""
        if data_dir is None:
            # Try to find data directory relative to script location
            script_dir = Path(__file__).parent
            if script_dir.name == 'data':
                self.data_dir = script_dir
            else:
                self.data_dir = script_dir / 'data'
        else:
            self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        self.analysis_results = {}
    
    def analyze_complete(self) -> Dict[str, Any]:
        """Perform complete analysis of all data and specifications."""
        print("🔍 Analyzing ACT framework data and specifications...")
        
        results = {
            "metadata": self._get_metadata(),
            "datasets": self._analyze_datasets(),
            "specifications": self._analyze_specifications(),
            "vnnlib_files": self._analyze_vnnlib_files(),
            "json_configs": self._analyze_json_configs(),
            "usage_recommendations": self._generate_usage_recommendations()
        }
        
        # Store results first, then compute statistics
        self.analysis_results = results
        
        # Now compute statistics with the complete results
        results["statistics"] = self._compute_statistics()
        
        return results
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data directory."""
        return {
            "data_directory": str(self.data_dir.absolute()),
            "total_files": len(list(self.data_dir.rglob("*"))),
            "total_size_mb": sum(f.stat().st_size for f in self.data_dir.rglob("*") if f.is_file()) / (1024 * 1024),
            "subdirectories": [d.name for d in self.data_dir.iterdir() if d.is_dir()],
            "file_extensions": list(set(f.suffix for f in self.data_dir.rglob("*") if f.is_file()))
        }
    
    def _analyze_datasets(self) -> Dict[str, Any]:
        """Analyze available datasets."""
        datasets = {}
        
        for subdir in self.data_dir.iterdir():
            if not subdir.is_dir() or subdir.name.startswith('.'):
                continue
            
            dataset_info = {
                "type": self._detect_dataset_type(subdir),
                "format": self._detect_format(subdir),
                "files": {}
            }
            
            for file_path in subdir.rglob("*"):
                if file_path.is_file():
                    file_info = {
                        "size_mb": file_path.stat().st_size / (1024 * 1024),
                        "description": self._get_file_description(file_path)
                    }
                    
                    # Analyze CSV files
                    if file_path.suffix.lower() == '.csv':
                        file_info["analysis"] = self._analyze_csv_file(file_path)
                    
                    dataset_info["files"][file_path.name] = file_info
            
            datasets[subdir.name.upper()] = dataset_info
        
        return datasets
    
    def _detect_dataset_type(self, path: Path) -> str:
        """Detect the type of dataset based on directory contents."""
        name = path.name.lower()
        if "mnist" in name:
            return "Image Classification"
        elif "cifar" in name:
            return "Image Classification"
        elif any(f.suffix == '.csv' for f in path.iterdir() if f.is_file()):
            return "CSV Dataset"
        else:
            return "Mixed Dataset"
    
    def _detect_format(self, path: Path) -> str:
        """Detect the primary format of files in the dataset."""
        extensions = [f.suffix.lower() for f in path.rglob("*") if f.is_file()]
        if '.csv' in extensions:
            return "CSV"
        elif any(ext in extensions for ext in ['.idx', '.idx1', '.idx3']):
            return "IDX Binary Format"
        elif '.json' in extensions:
            return "JSON"
        else:
            return "N/A"
    
    def _get_file_description(self, file_path: Path) -> str:
        """Generate description for a file based on its name and properties."""
        name = file_path.name.lower()
        size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if "train" in name and "images" in name:
            if size_mb > 10:
                return f"Training images ({int(size_mb*1024/44.86*60000):,} samples)" if "mnist" in str(file_path).lower() else f"Training images"
            else:
                return "Training images (compressed)"
        elif "train" in name and "labels" in name:
            return f"Training labels ({int(size_mb*1024/0.06*60000):,} labels)" if size_mb > 0.01 else "Training labels (compressed)"
        elif "test" in name or "t10k" in name:
            if "images" in name:
                return f"Test images ({int(size_mb*1024/7.48*10000):,} samples)" if size_mb > 1 else "Test images (compressed)"
            elif "labels" in name:
                return f"Test labels ({int(size_mb*1024/0.01*10000):,} labels)" if size_mb > 0.005 else "Test labels (compressed)"
        
        return "No description"
    
    def _analyze_csv_file(self, csv_file: Path) -> Dict[str, Any]:
        """Analyze individual CSV file with detailed data type and structure analysis."""
        if PANDAS_AVAILABLE:
            return self._analyze_csv_with_pandas(csv_file)
        else:
            return self._analyze_csv_basic(csv_file)
    
    def _analyze_csv_with_pandas(self, csv_file: Path) -> Dict[str, Any]:
        """Full CSV analysis using pandas."""
        try:
            # Read first few rows to understand structure
            df = pd.read_csv(csv_file, nrows=100)
            
            analysis = {
                "columns": list(df.columns),
                "num_columns": len(df.columns),
                "sample_count": len(df),
                "data_types": df.dtypes.to_dict(),
                "sample_data": df.head(3).to_dict('records') if len(df) > 0 else [],
                "data_structure": {},
                "verification_suitability": {}
            }
            
            # Get full file statistics
            try:
                full_df = pd.read_csv(csv_file)
                analysis["total_samples"] = len(full_df)
                
                # Detailed data structure analysis
                analysis["data_structure"] = self._analyze_data_structure(full_df, csv_file.name)
                
                # Identify label column and analyze
                label_info = self._analyze_labels(full_df)
                analysis.update(label_info)
                
                # Analyze input features
                feature_info = self._analyze_features(full_df, analysis.get("label_column"))
                analysis.update(feature_info)
                
                # Verification suitability assessment
                analysis["verification_suitability"] = self._assess_verification_suitability(
                    full_df, analysis["data_structure"]
                )
                
            except Exception as e:
                analysis["full_analysis_error"] = str(e)
                
            return analysis
            
        except Exception as e:
            return {"error": f"Failed to analyze CSV: {e}"}
    
    def _analyze_csv_basic(self, csv_file: Path) -> Dict[str, Any]:
        """Basic CSV analysis without pandas."""
        try:
            import csv
            
            with open(csv_file, 'r') as f:
                # Read first few rows to understand structure
                reader = csv.DictReader(f)
                rows = []
                for i, row in enumerate(reader):
                    rows.append(row)
                    if i >= 5:  # Read first 6 rows for analysis
                        break
            
            if not rows:
                return {"error": "Empty CSV file"}
            
            # Basic structure analysis
            first_row = rows[0]
            columns = list(first_row.keys())
            
            analysis = {
                "columns": columns,
                "num_columns": len(columns),
                "sample_data": rows[:1],  # First row only
                "data_structure": {},
                "verification_suitability": {}
            }
            
            # Count total rows (without reading all data)
            with open(csv_file, 'r') as f:
                total_rows = sum(1 for line in f) - 1  # Subtract header
                analysis["total_samples"] = total_rows
            
            # Basic structure detection
            if 'label' in columns and len(columns) == 785:  # MNIST structure
                analysis["data_structure"] = {
                    "dataset_type": "MNIST",
                    "input_format": "28×28 grayscale image",
                    "expected_shape": "28×28 grayscale (784 features)",
                    "pixel_count": 784,
                    "image_dimensions": "28×28",
                    "channels": 1
                }
                
                # Basic label analysis
                labels = []
                for row in rows:
                    if row['label'].isdigit():
                        labels.append(int(row['label']))
                
                if labels:
                    analysis["unique_labels"] = sorted(set(labels))
                    analysis["num_classes"] = len(set(labels))
                    analysis["label_range"] = f"[{min(labels)}-{max(labels)}]"
                
            elif 'label' in columns and len(columns) == 3073:  # CIFAR-10 structure
                analysis["data_structure"] = {
                    "dataset_type": "CIFAR-10",
                    "input_format": "32×32×3 color image",
                    "expected_shape": "32×32×3 RGB (3072 features)",
                    "pixel_count": 3072,
                    "image_dimensions": "32×32×3",
                    "channels": 3
                }
                
                # Basic label analysis for CIFAR
                labels = []
                for row in rows:
                    if row['label'].isdigit():
                        labels.append(int(row['label']))
                
                if labels:
                    analysis["unique_labels"] = sorted(set(labels))
                    analysis["num_classes"] = len(set(labels))
                    analysis["label_range"] = f"[{min(labels)}-{max(labels)}]"
            
            else:
                # Generic structure
                num_features = len(columns) - (1 if 'label' in columns else 0)
                analysis["data_structure"] = {
                    "dataset_type": "Custom",
                    "input_format": f"{num_features}-dimensional input",
                    "feature_count": num_features
                }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Failed to analyze CSV: {e}"}
    
    def _analyze_data_structure(self, df: Any, filename: str) -> Dict[str, Any]:
        """Analyze the structure and format of the data."""
        structure = {
            "dataset_type": "unknown",
            "input_format": "unknown",
            "expected_shape": "unknown",
            "normalization": "unknown"
        }
        
        # Detect dataset type based on filename and structure
        if "mnist" in filename.lower():
            structure["dataset_type"] = "MNIST"
            structure["input_format"] = "Image Classification"
            structure["expected_shape"] = "28×28 grayscale (784 features)"
            structure["normalization"] = "Typically [0,1] range"
            if len(df.columns) == 785:  # 784 pixels + 1 label
                structure["pixel_count"] = 784
                structure["image_dimensions"] = "28×28"
                structure["channels"] = 1
        elif "cifar" in filename.lower():
            structure["dataset_type"] = "CIFAR-10"
            structure["input_format"] = "Color Image Classification"
            structure["expected_shape"] = "32×32×3 RGB (3072 features)"
            structure["normalization"] = "Typically [0,1] or [-1,1] range"
            if len(df.columns) == 3073:  # 3072 pixels + 1 label
                structure["pixel_count"] = 3072
                structure["image_dimensions"] = "32×32×3"
                structure["channels"] = 3
        else:
            # Try to infer structure
            num_features = len(df.columns) - (1 if 'label' in df.columns else 0)
            structure["feature_count"] = num_features
            
            # Common image sizes
            if num_features == 784:
                structure["dataset_type"] = "MNIST-like"
                structure["input_format"] = "28×28 grayscale image"
            elif num_features == 3072:
                structure["dataset_type"] = "CIFAR-10-like"
                structure["input_format"] = "32×32×3 color image"
            else:
                structure["dataset_type"] = "Custom"
                structure["input_format"] = f"{num_features}-dimensional input"
        
        return structure
    
    def _analyze_labels(self, df: Any) -> Dict[str, Any]:
        """Analyze label structure and distribution."""
        label_info = {}
        
        # Find label column
        label_columns = [col for col in df.columns if 'label' in col.lower() or 'target' in col.lower() or 'y' == col.lower()]
        
        if label_columns:
            label_col = label_columns[0]
            label_info["label_column"] = label_col
            
            unique_labels = sorted(df[label_col].unique())
            label_info["unique_labels"] = unique_labels
            label_info["num_classes"] = len(unique_labels)
            label_info["label_range"] = f"[{min(unique_labels)}-{max(unique_labels)}]"
            label_info["label_distribution"] = df[label_col].value_counts().to_dict()
        
        return label_info
    
    def _analyze_features(self, df: Any, label_column: Optional[str] = None) -> Dict[str, Any]:
        """Analyze input features."""
        feature_info = {}
        
        # Get feature columns (exclude label)
        feature_cols = [col for col in df.columns if col != label_column]
        
        if feature_cols:
            feature_data = df[feature_cols]
            
            if NUMPY_AVAILABLE:
                feature_info["feature_analysis"] = {
                    "num_features": len(feature_cols),
                    "feature_statistics": {
                        "min": float(feature_data.min().min()),
                        "max": float(feature_data.max().max()),
                        "mean": float(feature_data.mean().mean()),
                        "std": float(feature_data.std().mean())
                    }
                }
        
        return feature_info
    
    def _assess_verification_suitability(self, df: Any, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Assess suitability for verification tasks."""
        suitability = {
            "recommended_specs": [],
            "suggested_epsilon": [],
            "difficulty_assessment": "unknown"
        }
        
        dataset_type = structure.get("dataset_type", "unknown").lower()
        
        if "mnist" in dataset_type:
            suitability["recommended_specs"] = ["local_lp", "local_vnnlib"]
            suitability["suggested_epsilon"] = [0.05, 0.1, 0.15, 0.2]
            suitability["difficulty_assessment"] = "beginner-friendly"
        elif "cifar" in dataset_type:
            suitability["recommended_specs"] = ["local_lp", "set_box"]
            suitability["suggested_epsilon"] = [8/255, 16/255]  # Common CIFAR values
            suitability["difficulty_assessment"] = "intermediate"
        else:
            suitability["recommended_specs"] = ["local_lp", "set_box"]
            suitability["suggested_epsilon"] = [0.1]
            suitability["difficulty_assessment"] = "depends on data range"
        
        return suitability
    
    def _analyze_specifications(self) -> Dict[str, Any]:
        """Analyze available specification types."""
        return {
            "LOCAL_LP": {
                "full_name": "Local Linear Programming",
                "description": "ε-ball around a specific input sample",
                "mathematical_form": "||x - x₀||_p ≤ ε",
                "use_cases": ["Adversarial robustness verification", "Local property checking around specific inputs", "Perturbation analysis"],
                "parameters": ["epsilon", "norm (inf/2/1)", "start/end sample indices"],
                "difficulty": "Easy to Medium"
            },
            "LOCAL_VNNLIB": {
                "full_name": "Local VNNLIB Specification",
                "description": "VNNLIB properties anchored to specific input points",
                "mathematical_form": "SMT-LIB constraints with fixed anchor variables",
                "use_cases": ["Custom local properties beyond simple ε-balls", "Competition benchmarks (VNN-COMP)", "Complex constraint combinations"],
                "parameters": ["vnnlib_path", "anchor points"],
                "difficulty": "Intermediate to Advanced"
            },
            "SET_VNNLIB": {
                "full_name": "Set-based VNNLIB Specification",
                "description": "Global properties over entire input domains",
                "mathematical_form": "∀x ∈ Domain: Property(f(x))",
                "use_cases": ["Safety verification over input ranges", "Global invariant checking", "Domain-wide property verification"],
                "parameters": ["vnnlib_path", "domain bounds"],
                "difficulty": "Advanced"
            },
            "SET_BOX": {
                "full_name": "Box Domain Specification",
                "description": "Verification over hyperrectangular input domains",
                "mathematical_form": "x ∈ [lb, ub]ⁿ",
                "use_cases": ["Domain verification with simple bounds", "Safety checking over operational ranges", "Global robustness analysis"],
                "parameters": ["input_lb", "input_ub"],
                "difficulty": "Intermediate"
            }
        }
    
    def _analyze_vnnlib_files(self) -> Dict[str, Any]:
        """Analyze VNNLIB specification files."""
        vnnlib_files = {}
        
        for vnnlib_file in self.data_dir.rglob("*.vnnlib"):
            try:
                file_info = {
                    "size_bytes": vnnlib_file.stat().st_size,
                    "analysis": self._parse_vnnlib_basic(vnnlib_file)
                }
                vnnlib_files[vnnlib_file.name] = file_info
            except Exception as e:
                vnnlib_files[vnnlib_file.name] = {"error": str(e)}
        
        return vnnlib_files
    
    def _parse_vnnlib_basic(self, vnnlib_file: Path) -> Dict[str, Any]:
        """Basic parsing of VNNLIB file to extract structure."""
        with open(vnnlib_file, 'r') as f:
            content = f.read()
        
        # Count variables and constraints
        input_vars = len([line for line in content.split('\n') if 'declare-const X_' in line])
        output_vars = len([line for line in content.split('\n') if 'declare-const Y_' in line])
        
        # Count constraints
        input_constraints = len([line for line in content.split('\n') if 'assert' in line and 'X_' in line])
        output_constraints = len([line for line in content.split('\n') if 'assert' in line and 'Y_' in line])
        
        analysis = {
            "num_input_vars": input_vars,
            "num_output_vars": output_vars,
            "num_input_constraints": input_constraints,
            "num_output_constraints": output_constraints
        }
        
        # Interpret overall specification
        interpretation = self._interpret_vnnlib_overall(analysis)
        analysis.update(interpretation)
        
        return analysis
    
    def _interpret_vnnlib_overall(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Provide high-level interpretation of the VNNLIB specification."""
        interpretation = {
            "specification_type": "local_vnnlib",  # Default to local_vnnlib
            "problem_description": "",
            "input_domain": {},
            "output_property": {},
            "verification_goal": ""
        }
        
        num_input_vars = analysis["num_input_vars"]
        num_output_vars = analysis["num_output_vars"]
        input_constraints = analysis["num_input_constraints"]
        output_constraints = analysis["num_output_constraints"]
        
        # Determine specification type based on documented types
        # Check constraint patterns to distinguish local_vnnlib vs set_vnnlib
        input_constraint_types = analysis.get("constraint_types", {}).get("input", {})
        
        # If we have many range constraints (pairs of upper/lower bounds), it's likely local_vnnlib
        # If we have fewer constraints relative to variables, it's likely set_vnnlib
        if input_constraints > 0 and output_constraints > 0:
            if input_constraints >= num_input_vars * 1.5:  # Many constraints suggest epsilon-ball (local)
                interpretation["specification_type"] = "local_vnnlib"
                interpretation["problem_description"] = "Local VNNLIB property anchored to specific input points"
            else:
                interpretation["specification_type"] = "set_vnnlib" 
                interpretation["problem_description"] = "Global VNNLIB property over input domain"
        elif input_constraints > 0:
            # Only input constraints, likely set_vnnlib
            interpretation["specification_type"] = "set_vnnlib"
            interpretation["problem_description"] = "Set-based VNNLIB specification with input domain constraints"
        else:
            # Default case
            interpretation["specification_type"] = "local_vnnlib"
            interpretation["problem_description"] = "VNNLIB specification (type unclear from constraints)"
        
        # Generate usage suggestion
        if num_input_vars == 784 and num_output_vars == 10:
            interpretation["usage_suggestion"] = "MNIST-like problem (784 inputs, 10 outputs) - suitable for image classification verification"
        elif num_input_vars == 3072 and num_output_vars == 10:
            interpretation["usage_suggestion"] = "CIFAR-10-like problem (3072 inputs, 10 outputs) - suitable for color image classification"
        else:
            interpretation["usage_suggestion"] = f"Custom problem with {num_input_vars} inputs and {num_output_vars} outputs"
        
        interpretation["command_type"] = interpretation["specification_type"]
        
        return interpretation
    
    def _analyze_json_configs(self) -> Dict[str, Any]:
        """Analyze JSON configuration files."""
        json_files = {}
        
        for json_file in self.data_dir.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    content = json.load(f)
                
                json_files[json_file.name] = {
                    "size_bytes": json_file.stat().st_size,
                    "structure": self._truncate_for_display(content),
                    "keys": list(content.keys()) if isinstance(content, dict) else "not_dict"
                }
            except Exception as e:
                json_files[json_file.name] = {"error": str(e)}
        
        return json_files
    
    def _truncate_for_display(self, data: Any) -> Any:
        """Truncate large data structures for display."""
        if isinstance(data, dict):
            return {k: (v if not isinstance(v, (dict, list)) else "...") 
                   for k, v in list(data.items())[:3]}
        elif isinstance(data, list):
            return data[:3] if len(data) <= 3 else data[:3] + ["..."]
        else:
            return data
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute overall statistics."""
        stats = {
            "summary": {
                "total_datasets": 0,
                "total_vnnlib_files": 0,
                "total_json_configs": 0,
                "largest_dataset_mb": 0,
                "supported_formats": ["IDX", "CSV", "VNNLIB", "JSON"]
            }
        }
        
        if self.analysis_results:
            datasets = self.analysis_results.get("datasets", {})
            stats["summary"]["total_datasets"] = len(datasets)
            
            vnnlib = self.analysis_results.get("vnnlib_files", {})
            stats["summary"]["total_vnnlib_files"] = len(vnnlib)
            
            json_configs = self.analysis_results.get("json_configs", {})
            stats["summary"]["total_json_configs"] = len(json_configs)
            
            # Find largest dataset
            max_size = 0
            for dataset_name, dataset_info in datasets.items():
                if "files" in dataset_info:
                    for file_info in dataset_info["files"].values():
                        size = file_info.get("size_mb", 0)
                        max_size = max(max_size, size)
            stats["summary"]["largest_dataset_mb"] = max_size
        
        return stats
    
    def _generate_usage_recommendations(self) -> Dict[str, Any]:
        """Generate practical usage recommendations."""
        return {
            "beginner_workflow": [
                "Start with MNIST dataset for simple image classification",
                "Use local_lp specification with small epsilon (0.1)",
                "Try different verifiers: eran, abcrown",
                "Gradually increase problem complexity"
            ],
            "command_examples": {
                "quick_analysis": "python data/analyze_data_and_specs.py",
                "mnist_verification": "python verifier/main.py --dataset mnist --model_path models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx --epsilon 0.1 --norm inf --start 0 --end 10"
            }
        }
    
    def analyze_specific_csv(self, csv_name: str) -> Dict[str, Any]:
        """Analyze a specific CSV file and provide specification recommendations."""
        csv_files = []
        for dataset_name, dataset_info in self.analysis_results.get("datasets", {}).items():
            for filename, file_info in dataset_info.get("files", {}).items():
                if filename.endswith('.csv') and csv_name.lower() in filename.lower():
                    csv_files.append((dataset_name, filename, file_info))
        
        if not csv_files:
            return {"error": f"CSV file containing '{csv_name}' not found"}
        
        # Use the first matching file
        dataset_name, filename, file_info = csv_files[0]
        analysis = file_info.get("analysis", {})
        
        if "error" in analysis:
            return {"error": f"Could not analyze {filename}: {analysis['error']}"}
        
        # Generate comprehensive specification recommendations
        recommendations = self._generate_csv_specifications(analysis, filename)
        
        return {
            "file_info": {
                "dataset": dataset_name,
                "filename": filename,
                "size_mb": file_info.get("size_mb", 0),
                "analysis": analysis
            },
            "recommendations": recommendations
        }
    
    def _generate_csv_specifications(self, analysis: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Generate specific input/output specification recommendations for a CSV file."""
        structure = analysis.get("data_structure", {})
        dataset_type = structure.get("dataset_type", "unknown")
        num_columns = analysis.get("num_columns", 0)
        unique_labels = analysis.get("unique_labels", [])
        num_classes = analysis.get("num_classes", 0)
        total_samples = analysis.get("total_samples", 0)
        
        recommendations = {
            "input_specifications": [],
            "output_specifications": [],
            "complete_commands": [],
            "python_api_examples": [],
            "difficulty_progression": [],
            "compatible_vnnlib": []
        }
        
        # Input specifications based on dataset type
        if "mnist" in dataset_type.lower():
            recommendations["input_specifications"] = [
                {
                    "type": "local_lp (L∞)",
                    "parameters": "ε=0.05",
                    "description": "Small perturbations (5% pixel change)",
                    "command": "--spec_type local_lp --norm inf --epsilon 0.05",
                    "difficulty": "Easy"
                },
                {
                    "type": "local_lp (L∞)",
                    "parameters": "ε=0.1", 
                    "description": "Medium perturbations (10% pixel change)",
                    "command": "--spec_type local_lp --norm inf --epsilon 0.1",
                    "difficulty": "Medium"
                },
                {
                    "type": "local_lp (L2)",
                    "parameters": "ε=1.0",
                    "description": "Euclidean distance constraint",
                    "command": "--spec_type local_lp --norm 2 --epsilon 1.0", 
                    "difficulty": "Medium"
                },
                {
                    "type": "local_vnnlib",
                    "parameters": "Custom constraints",
                    "description": "Maximum flexibility with VNNLIB",
                    "command": "--spec_type local_vnnlib --vnnlib_path data/vnnlib/set_vnnlib_example.vnnlib",
                    "difficulty": "Advanced"
                }
            ]
            
            # Check for compatible VNNLIB files
            vnnlib_files = self.analysis_results.get("vnnlib_files", {})
            for vnnlib_name, vnnlib_info in vnnlib_files.items():
                vnnlib_analysis = vnnlib_info.get("analysis", {})
                if vnnlib_analysis.get("num_input_vars") == 784:  # MNIST compatible
                    recommendations["compatible_vnnlib"].append({
                        "filename": vnnlib_name,
                        "input_vars": vnnlib_analysis.get("num_input_vars"),
                        "output_vars": vnnlib_analysis.get("num_output_vars"),
                        "description": vnnlib_analysis.get("problem_description", "")
                    })
        
        elif "cifar" in dataset_type.lower():
            recommendations["input_specifications"] = [
                {
                    "type": "local_lp (L∞)",
                    "parameters": "ε=8/255",
                    "description": "Standard CIFAR-10 perturbation",
                    "command": "--spec_type local_lp --norm inf --epsilon 0.031",
                    "difficulty": "Medium"
                },
                {
                    "type": "local_lp (L∞)",
                    "parameters": "ε=16/255", 
                    "description": "Larger CIFAR-10 perturbation",
                    "command": "--spec_type local_lp --norm inf --epsilon 0.063",
                    "difficulty": "Hard"
                }
            ]
        
        # Output specifications based on classification
        if num_classes > 1 and unique_labels:
            sample_label = unique_labels[0] if unique_labels else 0
            
            recommendations["output_specifications"] = [
                {
                    "type": "MARGIN_ROBUST",
                    "description": "Classification correctness (most common)",
                    "purpose": f"Ensures correct class prediction for all perturbations",
                    "example": f"For digit {sample_label}: y[{sample_label}] > max(other classes)"
                },
                {
                    "type": "TOP1_ROBUST", 
                    "description": "Argmax correctness (simplified)",
                    "purpose": "Checks if argmax(output) == true_class",
                    "example": f"argmax(y) == {sample_label}"
                },
                {
                    "type": "LINEAR_LE",
                    "description": "Custom constraints (advanced)",
                    "purpose": "Specific margin requirements between classes",
                    "example": f"y[{sample_label}] - y[{unique_labels[1] if len(unique_labels) > 1 else 0}] ≥ 0.1"
                }
            ]
        
        # Generate complete command examples
        if "mnist" in dataset_type.lower():
            model_path = "models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx"
            recommendations["complete_commands"] = [
                {
                    "name": "Basic Robustness Test",
                    "command": f"python -m act.wrapper_exts.ext_runner --dataset mnist --model_path {model_path} --spec_type local_lp --norm inf --epsilon 0.1 --start 0 --end 10 --verifier base"
                },
                {
                    "name": "Progressive Testing",
                    "command": "for eps in 0.05 0.1 0.15 0.2; do python -m act.wrapper_exts.ext_runner --dataset mnist --epsilon $eps --start 0 --end 10; done"
                }
            ]
            
            if recommendations["compatible_vnnlib"]:
                vnnlib_file = recommendations["compatible_vnnlib"][0]["filename"]
                recommendations["complete_commands"].append({
                    "name": "VNNLIB Test",
                    "command": f"python -m act.wrapper_exts.ext_runner --spec_type local_vnnlib --vnnlib_path data/vnnlib/{vnnlib_file} --model_path {model_path}"
                })
        
        # Difficulty progression
        recommendations["difficulty_progression"] = [
            {"level": "Beginner", "epsilon": "0.05", "expected": "Usually robust ✅"},
            {"level": "Standard", "epsilon": "0.1", "expected": "Sometimes robust ⚠️"},
            {"level": "Challenging", "epsilon": "0.15", "expected": "Rarely robust ❌"},
            {"level": "Stress Test", "epsilon": "0.2+", "expected": "Almost never robust ❌"}
        ]
        
        # Python API examples
        if unique_labels:
            sample_label = unique_labels[0]
            recommendations["python_api_examples"] = [
                {
                    "name": "L∞ Ball Input Specification",
                    "code": f"""
# Load sample from {filename}
sample_pixels = torch.tensor([...])  # {num_columns-1} values

input_spec = InputSpec(
    kind=InKind.LINF_BALL,
    center=sample_pixels,  # The CSV sample
    eps=0.1               # ±10% perturbation
)"""
                },
                {
                    "name": "Classification Output Specification", 
                    "code": f"""
output_spec = OutputSpec(
    kind=OutKind.MARGIN_ROBUST,
    y_true={sample_label},    # True class from CSV
    margin=0.0       # Minimum margin
)"""
                }
            ]
        
        return recommendations
    
    def print_analysis(self, focus_dataset: Optional[str] = None, focus_spec: Optional[str] = None, verbose: bool = False):
        """Print formatted analysis results."""
        if not self.analysis_results:
            self.analyze_complete()
        
        results = self.analysis_results
        
        print("\n" + "="*80)
        print("🎯 ACT FRAMEWORK DATA & SPECIFICATION ANALYSIS")
        print("="*80)
        
        # Metadata
        meta = results["metadata"]
        print(f"\n📊 Data Directory Overview:")
        print(f"   Location: {meta['data_directory']}")
        print(f"   Total Files: {meta['total_files']}")
        print(f"   Total Size: {meta['total_size_mb']:.2f} MB")
        print(f"   Subdirectories: {', '.join(meta['subdirectories'])}")
        
        # Datasets
        datasets = results["datasets"]
        print(f"\n📁 Available Datasets ({len(datasets)}):")
        
        for dataset_name, dataset_info in datasets.items():
            # Always show all datasets - no filtering
            print(f"\n   🔸 {dataset_name}")
            print(f"     Type: {dataset_info.get('type', 'Unknown')}")
            print(f"     Format: {dataset_info.get('format', 'N/A')}")
            
            # Show dataset structure information
            if 'data_structure' in dataset_info:
                struct = dataset_info['data_structure']
                if struct.get('dataset_type') != 'unknown':
                    print(f"     Dataset Type: {struct['dataset_type']}")
                    print(f"     Input Format: {struct['input_format']}")
                    print(f"     Expected Shape: {struct['expected_shape']}")
            
            if 'files' in dataset_info:
                print(f"     Files ({len(dataset_info['files'])}):")
                for filename, file_info in dataset_info['files'].items():
                    size_mb = file_info.get('size_mb', 0)
                    desc = file_info.get('description', 'No description')
                    print(f"       • {filename} ({size_mb:.2f} MB) - {desc}")
                    
                    # Show CSV structure details immediately
                    if 'analysis' in file_info and filename.endswith('.csv'):
                        analysis = file_info['analysis']
                        if 'error' not in analysis:
                            print(f"         📊 CSV Structure:")
                            print(f"         Rows: {analysis.get('total_samples', 'Unknown')}")
                            print(f"         Columns: {analysis.get('num_columns', 'Unknown')}")
                            if 'columns' in analysis:
                                cols = analysis['columns']
                                if len(cols) <= 5:
                                    print(f"         Column Names: {', '.join(cols)}")
                                else:
                                    print(f"         Column Names: {', '.join(cols[:3])}, ... (showing first 3 of {len(cols)})")
                            
                            # Show label information
                            if 'num_classes' in analysis:
                                print(f"         Labels: {analysis['num_classes']} classes {analysis.get('label_range', '')}")
                                if 'unique_labels' in analysis:
                                    labels = analysis['unique_labels']
                                    if len(labels) <= 10:
                                        print(f"         Label Values: {labels}")
                                    else:
                                        print(f"         Label Values: {labels[:5]}... (showing first 5)")
                            
                            # Show sample data
                            if 'sample_data' in analysis and analysis['sample_data']:
                                print(f"         Sample Data (first row):")
                                sample = analysis['sample_data'][0]
                                sample_items = list(sample.items())[:5]  # Show first 5 columns
                                for key, value in sample_items:
                                    if isinstance(value, float):
                                        print(f"           {key}: {value:.3f}")
                                    else:
                                        print(f"           {key}: {value}")
                                if len(sample) > 5:
                                    print(f"           ... (showing first 5 of {len(sample)} columns)")
                    
                    if verbose and 'analysis' in file_info:
                        analysis = file_info['analysis']
                        
                        # Show data structure details
                        if 'data_structure' in analysis:
                            struct = analysis['data_structure']
                            print(f"         Data Type: {struct.get('dataset_type', 'Unknown')}")
                            if 'pixel_count' in struct:
                                print(f"         Pixel Count: {struct['pixel_count']}")
                            if 'image_dimensions' in struct:
                                print(f"         Image Dimensions: {struct['image_dimensions']}")
                        
                        # Show feature analysis
                        if 'feature_analysis' in analysis:
                            feat = analysis['feature_analysis']
                            if 'feature_statistics' in feat:
                                stats = feat['feature_statistics']
                                print(f"         Feature Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                                print(f"         Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        
        # Specifications
        specs = results["specifications"]
        print(f"\n📋 Specification Types:")
        
        for spec_type, spec_info in specs.items():
            if focus_spec and focus_spec.lower() not in spec_type.lower():
                continue
                
            print(f"\n   🔹 {spec_type}")
            print(f"     Full Name: {spec_info['full_name']}")
            print(f"     Description: {spec_info['description']}")
            print(f"     Mathematical Form: {spec_info['mathematical_form']}")
            print(f"     Use Cases: {', '.join(spec_info['use_cases'][:2])}...")
        
        # VNNLIB files
        vnnlib_files = results["vnnlib_files"]
        if vnnlib_files:
            print(f"\n📄 VNNLIB Files ({len(vnnlib_files)}):")
            
            for filename, file_info in vnnlib_files.items():
                print(f"\n   🔸 {filename}")
                print(f"     Size: {file_info.get('size_bytes', 0)} bytes")
                
                if 'analysis' in file_info:
                    analysis = file_info['analysis']
                    print(f"     Input Variables: {analysis.get('num_input_vars', 'Unknown')}")
                    print(f"     Output Variables: {analysis.get('num_output_vars', 'Unknown')}")
                    print(f"     Input Constraints: {analysis.get('num_input_constraints', 'Unknown')}")
                    print(f"     Output Constraints: {analysis.get('num_output_constraints', 'Unknown')}")
                    print(f"     Specification Type: {analysis.get('specification_type', 'Unknown')}")
                    print(f"     Problem: {analysis.get('problem_description', 'Unknown')}")
                    if 'usage_suggestion' in analysis:
                        print(f"     Usage: {analysis['usage_suggestion']}")
                    print(f"     Command Type: {analysis.get('command_type', 'Unknown')}")
        
        # JSON configs
        if verbose and not focus_spec:
            json_files = results["json_configs"]
            if json_files:
                print(f"\n📄 JSON Configuration Files ({len(json_files)}):")
                
                for filename, file_info in json_files.items():
                    print(f"\n   🔸 {filename}")
                    print(f"     Size: {file_info.get('size_bytes', 0)} bytes")
                    if 'keys' in file_info and file_info['keys'] != 'not_dict':
                        print(f"     Keys: {', '.join(file_info['keys'])}")
        
        # Usage recommendations
        if verbose and not focus_spec:
            recommendations = results["usage_recommendations"]
            print(f"\n🚀 Usage Recommendations:")
            
            print(f"\n   Beginner Workflow:")
            for i, step in enumerate(recommendations["beginner_workflow"], 1):
                print(f"     {i}. {step}")
            
            print(f"\n   Command Examples:")
            for name, command in recommendations["command_examples"].items():
                print(f"     {name.replace('_', ' ').title()}: {command}")
        
        # Statistics
        stats = results["statistics"]["summary"]
        print(f"\n📈 Summary Statistics:")
        print(f"   Total Datasets: {stats['total_datasets']}")
        print(f"   Total VNNLIB Files: {stats['total_vnnlib_files']}")
        print(f"   Total JSON Configs: {stats['total_json_configs']}")
        print(f"   Largest Dataset: {stats['largest_dataset_mb']:.2f} MB")
        print(f"   Supported Formats: {', '.join(stats['supported_formats'])}")
        
        print("\n" + "="*80)
        print("💡 Tip: Use --verbose for detailed analysis, --csv <name> for CSV-specific recommendations")
        print("="*80)
    
    def print_csv_analysis(self, csv_name: str):
        """Print detailed analysis and recommendations for a specific CSV file."""
        if not self.analysis_results:
            self.analyze_complete()
        
        result = self.analyze_specific_csv(csv_name)
        
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        
        file_info = result["file_info"]
        recommendations = result["recommendations"]
        
        print("\n" + "="*80)
        print(f"🎯 CSV SPECIFICATION ANALYSIS: {file_info['filename']}")
        print("="*80)
        
        # File information
        analysis = file_info["analysis"]
        structure = analysis.get("data_structure", {})
        
        print(f"\n📊 File Overview:")
        print(f"   Dataset: {file_info['dataset']}")
        print(f"   File: {file_info['filename']} ({file_info['size_mb']:.2f} MB)")
        print(f"   Rows: {analysis.get('total_samples', 'Unknown')}")
        print(f"   Columns: {analysis.get('num_columns', 'Unknown')}")
        print(f"   Dataset Type: {structure.get('dataset_type', 'Unknown')}")
        print(f"   Input Format: {structure.get('input_format', 'Unknown')}")
        
        if analysis.get("unique_labels"):
            labels = analysis["unique_labels"]
            print(f"   Labels: {analysis.get('num_classes', 0)} classes {analysis.get('label_range', '')}")
            if len(labels) <= 10:
                print(f"   Label Values: {labels}")
            else:
                print(f"   Label Values: {labels[:5]}... (first 5 of {len(labels)})")
        
        # Input specifications
        print(f"\n🎯 Recommended Input Specifications:")
        print("-" * 60)
        print(f"{'Type':<20} {'Parameters':<12} {'Description':<25} {'Difficulty':<10}")
        print("-" * 60)
        for spec in recommendations["input_specifications"]:
            print(f"{spec['type']:<20} {spec['parameters']:<12} {spec['description']:<25} {spec['difficulty']:<10}")
        
        # Output specifications
        print(f"\n🎯 Recommended Output Specifications:")
        print("-" * 60)
        for spec in recommendations["output_specifications"]:
            print(f"   • {spec['type']}: {spec['description']}")
            print(f"     Purpose: {spec['purpose']}")
            print(f"     Example: {spec['example']}")
            print()
        
        # Compatible VNNLIB files
        if recommendations["compatible_vnnlib"]:
            print(f"🎯 Compatible VNNLIB Files:")
            for vnnlib in recommendations["compatible_vnnlib"]:
                print(f"   • {vnnlib['filename']}")
                print(f"     Input Vars: {vnnlib['input_vars']}, Output Vars: {vnnlib['output_vars']}")
                print(f"     Description: {vnnlib['description']}")
        
        # Complete commands
        print(f"\n🚀 Complete Command Examples:")
        for cmd in recommendations["complete_commands"]:
            print(f"\n   {cmd['name']}:")
            print(f"   {cmd['command']}")
        
        # Python API examples
        if recommendations["python_api_examples"]:
            print(f"\n🐍 Python API Examples:")
            for example in recommendations["python_api_examples"]:
                print(f"\n   {example['name']}:")
                print(example['code'])
        
        # Difficulty progression
        print(f"\n🎯 Difficulty Progression:")
        print("-" * 50)
        for prog in recommendations["difficulty_progression"]:
            print(f"   {prog['level']:<12}: ε={prog['epsilon']:<6} → {prog['expected']}")
        
        print("\n" + "="*80)
        print("💡 Start with small ε values and increase gradually")
        print("💡 Use --verbose with main analysis for more dataset details")
        print("="*80)
    
    def print_specification_matrix(self):
        """Print specification compatibility matrix."""
        print("\n" + "="*80)
        print("🎯 SPECIFICATION COMPATIBILITY MATRIX")
        print("="*80)
        
        # Input specifications
        input_specs = [
            ("local_lp (L∞)", "ε=0.05", "Small perturbations", "Easy verification"),
            ("local_lp (L∞)", "ε=0.1", "Medium perturbations", "Medium difficulty"),
            ("local_lp (L∞)", "ε=0.2", "Large perturbations", "Hard verification"),
            ("local_lp (L2)", "ε=1.0", "Euclidean distance", "Natural for images"),
            ("local_lp (L1)", "ε=10.0", "Sparse perturbations", "Rare pixels change"),
            ("local_vnnlib", "Custom", "VNNLIB constraints", "Maximum flexibility"),
            ("set_box", "[0,1]^n", "Global bounds", "Domain verification")
        ]
        
        # Output specifications  
        output_specs = [
            ("MARGIN_ROBUST", "Classification correctness", "Most common"),
            ("TOP1_ROBUST", "Argmax correctness", "Simplified version"),
            ("LINEAR_LE", "Custom constraints", "Advanced properties")
        ]
        
        print("\n📊 Input Specifications:")
        print("-" * 80)
        print(f"{'Type':<15} {'Parameter':<10} {'Description':<20} {'Difficulty':<15}")
        print("-" * 80)
        for spec_type, param, desc, diff in input_specs:
            print(f"{spec_type:<15} {param:<10} {desc:<20} {diff:<15}")
        
        print("\n📊 Output Specifications:")
        print("-" * 60)
        print(f"{'Type':<15} {'Purpose':<25} {'Usage':<15}")
        print("-" * 60)
        for spec_type, purpose, usage in output_specs:
            print(f"{spec_type:<15} {purpose:<25} {usage:<15}")
        
        print("\n🎯 Recommended Combinations:")
        print("-" * 80)
        
        combinations = [
            ("Beginner", "local_lp + MARGIN_ROBUST", "ε=0.05, test 5 samples"),
            ("Standard", "local_lp + MARGIN_ROBUST", "ε=0.1, test 10-20 samples"),  
            ("Advanced", "local_lp + LINEAR_LE", "ε=0.15, custom margins"),
            ("Research", "local_vnnlib + MARGIN_ROBUST", "Complex constraints"),
            ("Stress Test", "local_lp + MARGIN_ROBUST", "ε=0.3, expect failures")
        ]
        
        for level, combination, details in combinations:
            print(f"{level:<12}: {combination:<25} ({details})")
        
        print("\n💡 Key Insights:")
        print("   • Most models robust at ε≤0.05, few at ε≥0.2")
        print("   • L∞ norm most common, L2 more natural for images")
        print("   • Start with MARGIN_ROBUST, move to LINEAR_LE for custom properties")
        print("   • VNNLIB provides maximum flexibility")
        print("="*80)
    
    def save_analysis(self, output_file: Optional[str] = None):
        """Save analysis results to JSON file."""
        if not self.analysis_results:
            self.analyze_complete()
        
        if output_file is None:
            output_file = "data_spec_analysis.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            print(f"📁 Analysis saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="ACT Framework Comprehensive Data & Specification Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete analysis + specification matrix (default)
  python analyze_data_spec.py
  
  # Detailed analysis with feature statistics + matrix
  python analyze_data_spec.py --verbose
  
  # Focus on specific CSV file with recommendations  
  python analyze_data_spec.py --csv mnist
  python analyze_data_spec.py --csv cifar
  
  # Show only specification compatibility matrix
  python analyze_data_spec.py --matrix
  
  # Focus on specific dataset (no matrix at end)
  python analyze_data_spec.py mnist --verbose
  
  # Focus on specifications (no matrix at end)
  python analyze_data_spec.py --spec vnnlib
  
  # Detailed analysis with JSON output
  python analyze_data_spec.py --verbose --save-json
  
  # Custom data directory
  python analyze_data_spec.py --data-dir /path/to/data
        """
    )
    
    parser.add_argument("dataset", nargs="?", type=str, help="Enhanced focus on specific dataset (shows all datasets with extra details for specified one)")
    parser.add_argument("--dataset", dest="dataset_flag", type=str, help="Enhanced focus on specific dataset (alternative to positional argument)")
    parser.add_argument("--csv", type=str, help="Analyze specific CSV file and provide specification recommendations")
    parser.add_argument("--spec", type=str, help="Focus analysis on specific specification type")
    parser.add_argument("--matrix", action="store_true", help="Show only specification compatibility matrix")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed dataset and specification analysis")
    parser.add_argument("--save-json", action="store_true", help="Save analysis to JSON file")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file name")
    parser.add_argument("--data-dir", type=str, help="Custom data directory path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output, only save JSON")
    
    args = parser.parse_args()
    
    # Handle dataset argument (positional or flag)
    dataset_focus = args.dataset or args.dataset_flag
    
    # Check dependencies
    if not NUMPY_AVAILABLE and not args.quiet:
        print("⚠️  NumPy not available. Some analysis features will be limited.")
        print("   Install with: pip install numpy")
        
    if not PANDAS_AVAILABLE and not args.quiet:
        print("⚠️  Pandas not available. CSV analysis will be limited.")
        print("   Install with: pip install pandas")
    
    try:
        # Create analyzer
        analyzer = ComprehensiveAnalyzer(args.data_dir)
        
        # Perform analysis
        analyzer.analyze_complete()
        
        # Handle different output modes
        if args.csv:
            analyzer.print_csv_analysis(args.csv)
        elif args.matrix:
            # Show only matrix if explicitly requested
            analyzer.print_specification_matrix()
        elif not args.quiet:
            # Always show detailed analysis first (unless --csv or --matrix only)
            analyzer.print_analysis(
                focus_dataset=dataset_focus,
                focus_spec=args.spec,
                verbose=args.verbose
            )
            
            # Then show specification matrix at the end (unless specific focus is requested)
            if not dataset_focus and not args.spec:
                print("\n" + "="*80)
                print("📋 SPECIFICATION COMPATIBILITY REFERENCE")
                print("="*80)
                analyzer.print_specification_matrix()
        
        # Save to JSON if requested
        if args.save_json or args.output:
            analyzer.save_analysis(args.output)
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()