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
        # Determine best available device
        if torch.cuda.is_available():
            try:
                # Test CUDA device
                test_device = torch.device("cuda:0")
                test_tensor = torch.zeros(1, device=test_device)
                del test_tensor  # Clean up
                target_device = test_device
                print(f"Successfully initialized CUDA device: cuda:0")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"CUDA device cuda:0 error ({e}), falling back to CPU")
                target_device = torch.device("cpu")
        else:
            target_device = torch.device("cpu")
        
        # Set PyTorch global defaults
        torch.set_default_dtype(torch.float64)
        if hasattr(torch, 'set_default_device'):
            try:
                torch.set_default_device(target_device)
                print(f"✅ Initialized: device={target_device}, dtype=torch.float64")
            except Exception as e:
                print(f"✅ Initialized: dtype=torch.float64 (device setting not supported: {e})")
        else:
            print(f"✅ Initialized: dtype=torch.float64 (device setting not available in this PyTorch version)")
            
    except Exception as e:
        # Last resort fallback - just set dtype
        print(f"Device initialization failed ({e}), using CPU + float64")
        torch.set_default_dtype(torch.float64)
    
    finally:
        # Mark as initialized regardless of success/failure
        _INITIALIZED = True

# Auto-initialize when module is imported
_auto_initialize()
