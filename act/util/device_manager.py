# device_manager.py
# Simplified device/dtype management using PyTorch global defaults.

import torch
from typing import Tuple

# Global initialization state
_INITIALIZED = False

def get_default_device() -> torch.device:
    """Get current PyTorch default device."""
    _auto_initialize()  # Ensure initialization before returning
    
    if hasattr(torch, 'get_default_device'):
        try:
            return torch.get_default_device()
        except:
            return torch.device("cpu")  # Fallback
    else:
        # For older PyTorch versions, check where a test tensor is created
        test_tensor = torch.zeros(1)
        device = test_tensor.device
        del test_tensor
        return device

def get_default_dtype() -> torch.dtype:
    """Get current PyTorch default dtype."""
    _auto_initialize()  # Ensure initialization before returning
    return torch.get_default_dtype()

def get_current_settings() -> Tuple[torch.device, torch.dtype]:
    """Get current PyTorch default device and dtype settings."""
    _auto_initialize()  # Ensure initialization before returning
    return get_default_device(), get_default_dtype()

# Auto-initialize with sensible defaults when module is imported
def _auto_initialize():
    """Automatic initialization with sensible defaults."""
    global _INITIALIZED
    
    # Skip if already initialized
    if _INITIALIZED:
        return
    
    try:
        # Get device and dtype preferences from command line arguments
        preferred_device = 'cuda'  # Default value from options.py
        preferred_dtype = 'float64'  # Default value from options.py
        try:
            import sys
            from act.util.options import get_parser
            
            parser = get_parser()
            args, _ = parser.parse_known_args(sys.argv[1:])
            preferred_device = args.device
            preferred_dtype = args.dtype
            
            # Handle gpu/cuda aliasing
            if preferred_device == 'gpu':
                preferred_device = 'cuda'
                print(f"🔄 Device alias: 'gpu' → 'cuda'")
                
        except Exception:
            # If parsing fails, use default
            pass
        
        # Initialize device
        if preferred_device == 'cpu':
            target_device = torch.device("cpu")
            print(f"Using command line device: cpu")
        else:  # cuda or any other value
            if torch.cuda.is_available():
                target_device = torch.device("cuda:0")
                print(f"Using command line device: cuda:0")
            else:
                target_device = torch.device("cpu")
                print(f"CUDA not available, using CPU")
        
        # Set PyTorch global defaults
        target_dtype = torch.float64 if preferred_dtype == 'float64' else torch.float32
        torch.set_default_dtype(target_dtype)
        if hasattr(torch, 'set_default_device'):
            torch.set_default_device(target_device)
        print(f"✅ Initialized: device={target_device}, dtype={target_dtype}")
            
    except Exception as e:
        # Last resort fallback
        print(f"Device initialization failed ({e}), using CPU + float64")
        torch.set_default_dtype(torch.float64)
    
    finally:
        # Mark as initialized regardless of success/failure
        _INITIALIZED = True

# Auto-initialize when module is imported
_auto_initialize()
