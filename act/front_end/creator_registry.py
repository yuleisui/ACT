#!/usr/bin/env python3
"""
Creator Registry for ACT Specification Creators.

Provides factory pattern for managing multiple spec creators (TorchVision, VNNLIB, bert)
with automatic detection and routing.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import logging
from typing import Dict, Tuple, Optional, List
from act.front_end.spec_creator_base import BaseSpecCreator
from act.util.format_utils import rule

logger = logging.getLogger(__name__)


class CreatorRegistry:
    """
    Registry for managing specification creators.
    
    Provides factory methods for accessing TorchVision and VNNLIB creators,
    with automatic detection based on dataset/category names.
    """
    
    _creators: Dict[str, BaseSpecCreator] = {}
    
    @classmethod
    def register(cls, name: str, creator: BaseSpecCreator) -> None:
        """
        Register a spec creator.
        
        Args:
            name: Creator name ('torchvision' or 'vnnlib')
            creator: Creator instance
        """
        cls._creators[name] = creator
    
    @classmethod
    def get_creator(cls, name: str) -> BaseSpecCreator:
        """
        Get a spec creator by name.
        
        Args:
            name: Creator name ('torchvision' or 'vnnlib')
            
        Returns:
            Creator instance
            
        Raises:
            ValueError: If creator not found
        """
        if name not in cls._creators:
            # Lazy initialization
            if name == 'torchvision':
                from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator
                cls._creators[name] = TorchVisionSpecCreator()
            elif name == 'vnnlib':
                from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
                cls._creators[name] = VNNLibSpecCreator()
            elif name == 'bert':
                from act.front_end.bert_loader.create_specs import BertSpecCreator
                cls._creators['bert'] = BertSpecCreator()
            else:
                raise ValueError(
                    f"Unknown creator '{name}'. "
                    f"Available creators: {list_creators()}"
                )
        
        return cls._creators[name]
    
    @classmethod
    def list_creators(cls) -> List[str]:
        """
        List all available creator names.
        
        Returns:
            List of creator names
        """
        return ['torchvision', 'vnnlib', 'bert']
    
    @classmethod
    def detect_creator(cls, name: str, explicit_creator: Optional[str] = None) -> Tuple[str, str]:
        """
        Auto-detect which creator to use based on name.
        
        Tries to match the name against TorchVision datasets and VNNLIB categories.
        If explicit_creator is provided, skips detection and validates the name
        against that creator only.
        
        Args:
            name: Dataset or category name (case-insensitive)
            explicit_creator: Optional creator name to force ('torchvision' or 'vnnlib')
            
        Returns:
            Tuple of (creator_name, normalized_name)
            - creator_name: 'torchvision' or 'vnnlib'
            - normalized_name: Properly cased name for the creator
            
        Raises:
            ValueError: If name not found, ambiguous, or invalid for explicit creator
            
        Examples:
            >>> detect_creator('MNIST')
            ('torchvision', 'MNIST')
            
            >>> detect_creator('acasxu_2023')
            ('vnnlib', 'acasxu_2023')
            
            >>> detect_creator('mnist', 'vnnlib')
            ValueError: 'mnist' not found in VNNLIB categories
        """
        # If explicit creator specified, only check that one
        if explicit_creator:
            if explicit_creator not in cls.list_creators():
                raise ValueError(
                    f"Unknown creator '{explicit_creator}'. "
                    f"Available: {cls.list_creators()}"
                )
            
            if explicit_creator == 'torchvision':
                from act.front_end.torchvision_loader.data_model_mapping import find_dataset_name
                try:
                    normalized = find_dataset_name(name)
                    return ('torchvision', normalized)
                except ValueError as e:
                    raise ValueError(
                        f"Dataset '{name}' not found in TorchVision creator.\n{str(e)}"
                    )
            elif explicit_creator == 'vnnlib':
                from act.front_end.vnnlib_loader.category_mapping import find_category_name
                try:
                    normalized = find_category_name(name)
                    return ('vnnlib', normalized)
                except ValueError as e:
                    raise ValueError(
                        f"Category '{name}' not found in VNNLIB creator.\n{str(e)}"
                    )
            else:
                from act.front_end.bert_loader.data_loader import find_bert_dataset_name
                try:
                    normalized = find_bert_dataset_name(name)
                    return ('bert', normalized)
                except ValueError as e:
                    raise ValueError(
                        f"Dataset '{name}' not found in bert creator.\n{str(e)}"
                    )
        
        # Auto-detection: try both creators
        torchvision_match = False
        vnnlib_match = False
        bert_match = False
        tv_name = None
        vnnlib_name = None
        bert_name = None
        
        # Try TorchVision
        try:
            from act.front_end.torchvision_loader.data_model_mapping import find_dataset_name
            tv_name = find_dataset_name(name)
            torchvision_match = True
        except ValueError as e:
            # Intentional: auto-detection probe; name not in TorchVision is normal, try VNNLIB next.
            logger.debug("suppressed: %s", e)
        
        # Try VNNLIB
        try:
            from act.front_end.vnnlib_loader.category_mapping import find_category_name
            vnnlib_name = find_category_name(name)
            vnnlib_match = True
        except ValueError as e:
            # Intentional: auto-detection probe; absence in VNNLIB is reported via the match flags below.
            logger.debug("suppressed: %s", e)

        try:
            from act.front_end.bert_loader.data_loader import find_bert_dataset_name
            bert_name = find_bert_dataset_name(name)
            bert_match = True
        except ValueError as e:
            logger.debug("suppressed: %s", e)
        
        # Handle results
        if sum([torchvision_match, vnnlib_match, bert_match]) > 1:
            raise ValueError(
                f"Ambiguous name '{name}' matches multiple creators:\n"
                f"  TorchVision: {tv_name}\n"
                f"  VNNLIB: {vnnlib_name}\n"
                f"  BERT: {bert_name}\n"
                f"Use --creator to specify explicitly:\n"
                f"  --creator torchvision  OR  --creator vnnlib  OR  --creator bert"
            )
        elif torchvision_match:
            assert tv_name is not None
            return ('torchvision', tv_name)
        elif vnnlib_match:
            assert vnnlib_name is not None
            return ('vnnlib', vnnlib_name)
        elif bert_match:
            assert bert_name is not None
            return ('bert', bert_name)
        else:
            raise ValueError(
                f"Dataset/category '{name}' not found in any creator.\n"
                f"Use --list to see all available options."
            )


# Convenience functions for external use
def get_creator(name: str) -> BaseSpecCreator:
    """Get a spec creator by name."""
    return CreatorRegistry.get_creator(name)


def list_creators() -> List[str]:
    """List all available creator names."""
    return CreatorRegistry.list_creators()


def detect_creator(name: str, explicit_creator: Optional[str] = None) -> Tuple[str, str]:
    """Auto-detect which creator to use based on name."""
    return CreatorRegistry.detect_creator(name, explicit_creator)


if __name__ == "__main__":
    # Quick demo
    print(rule())
    print("CREATOR REGISTRY DEMO")
    print(rule())
    
    print("\nAvailable creators:", list_creators())
    
    # Test auto-detection
    print("\n" + rule())
    print("AUTO-DETECTION TESTS")
    print(rule())
    
    test_cases = [
        "MNIST",
        "mnist",
        "acasxu_2023",
        "CIFAR10",
        "vggnet16_2022",
    ]
    
    for test in test_cases:
        try:
            creator_name, normalized = detect_creator(test)
            print(f"\n'{test}' → {creator_name}: {normalized}")
        except ValueError as e:
            print(f"\n'{test}' → ERROR: {e}")
