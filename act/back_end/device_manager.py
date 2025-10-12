# device_manager.py
# Simplified device/dtype management using PyTorch global defaults.

import torch
import contextlib
from typing import Optional, Tuple

# --- Global initialization state ---
_INITIALIZED = False

def initialize_device_dtype(device_arg: str = "cuda", dtype_arg: str = "float64") -> Tuple[torch.device, torch.dtype]:
    """
    Initialize device and dtype from command-line arguments.
    Sets PyTorch global defaults so all tensor creation uses correct device/dtype automatically.
    
    Args:
        device_arg: Device specification ('cpu', 'cuda', 'cuda:0', etc.)
        dtype_arg: Dtype specification ('float16', 'float32', 'float64')
    
    Returns:
        tuple: (actual_device, actual_dtype) that were set
    """
    global _INITIALIZED
    
    # Parse dtype argument
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32, 
        'float64': torch.float64
    }
    torch_dtype = dtype_map.get(dtype_arg, torch.float64)
    
    # Determine target device with fallback logic
    if device_arg == "cuda":
        device_arg = "cuda:0"  # Default to first GPU
    
    target_device = torch.device("cpu")  # Default fallback
    
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"CUDA not available, using CPU instead of {device_arg}")
            target_device = torch.device("cpu")
        else:
            try:
                test_device = torch.device(device_arg)
                # Test if device is actually usable
                test_tensor = torch.zeros(1, device=test_device)
                del test_tensor  # Clean up
                target_device = test_device
                print(f"Successfully initialized CUDA device: {device_arg}")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"CUDA device {device_arg} error ({e}), falling back to CPU")
                target_device = torch.device("cpu")
    elif device_arg == "cpu":
        target_device = torch.device("cpu")
    else:
        print(f"Unknown device specification '{device_arg}', using CPU")
        target_device = torch.device("cpu")
    
    # Set PyTorch global defaults
    torch.set_default_dtype(torch_dtype)
    if hasattr(torch, 'set_default_device'):
        try:
            torch.set_default_device(target_device)
            print(f"✅ Initialized: device={target_device}, dtype={torch_dtype}")
        except Exception as e:
            print(f"✅ Initialized: dtype={torch_dtype} (device setting not supported: {e})")
    else:
        print(f"✅ Initialized: dtype={torch_dtype} (device setting not available in this PyTorch version)")
    
    _INITIALIZED = True
    return target_device, torch_dtype

def ensure_initialized(device: str = "cuda", dtype: str = "float64") -> Tuple[torch.device, torch.dtype]:
    """
    Ensure ACT is initialized. Safe to call multiple times.
    
    Args:
        device: Default device if not already initialized
        dtype: Default dtype if not already initialized
    
    Returns:
        tuple: Current (device, dtype) settings
    """
    if not _INITIALIZED:
        return initialize_device_dtype(device, dtype)
    else:
        return get_current_settings()

def get_current_settings() -> Tuple[torch.device, torch.dtype]:
    """
    Get current PyTorch default settings.
    
    Returns:
        tuple: (current_device, current_dtype)
    """
    current_dtype = torch.get_default_dtype()
    
    current_device = None
    if hasattr(torch, 'get_default_device'):
        try:
            current_device = torch.get_default_device()
        except:
            current_device = torch.device("cpu")  # Fallback
    else:
        # For older PyTorch versions, check where a test tensor is created
        test_tensor = torch.zeros(1)
        current_device = test_tensor.device
        del test_tensor
    
    return current_device, current_dtype

@contextlib.contextmanager
def temp_device_dtype(device: Optional[str] = None, dtype: Optional[str] = None):
    """
    Temporarily change PyTorch default device/dtype within context.
    
    Args:
        device: Temporary device ('cpu', 'cuda', etc.)
        dtype: Temporary dtype ('float16', 'float32', 'float64')
    """
    # Store original settings
    original_device, original_dtype = get_current_settings()
    
    try:
        # Set temporary settings if provided
        if device is not None or dtype is not None:
            temp_device_arg = device or (original_device.type if original_device else "cpu")
            temp_dtype_arg = dtype or {
                torch.float16: "float16",
                torch.float32: "float32", 
                torch.float64: "float64"
            }.get(original_dtype, "float64")
            
            initialize_device_dtype(temp_device_arg, temp_dtype_arg)
        
        yield
        
    finally:
        # Restore original settings
        if original_device and original_dtype:
            torch.set_default_dtype(original_dtype)
            if hasattr(torch, 'set_default_device'):
                try:
                    torch.set_default_device(original_device)
                except:
                    pass

def summary() -> str:
    """Get current ACT status summary."""
    if not _INITIALIZED:
        return "ACT not initialized - call initialize_device_dtype() or ensure_initialized()"
    
    device, dtype = get_current_settings()
    cuda_available = torch.cuda.is_available()
    
    return f"device={device}, dtype={dtype}, cuda_available={cuda_available}, initialized={_INITIALIZED}"

def wrap_model_fn(module: torch.nn.Module):
    """Convert PyTorch module to use current default device/dtype."""
    device, dtype = get_current_settings()
    mod = module.to(device=device, dtype=dtype)
    
    @torch.no_grad()
    def f(x: torch.Tensor) -> torch.Tensor:
        return mod(x.to(device=device, dtype=dtype))
    return f

# Auto-initialize with sensible defaults when module is imported
def _auto_initialize():
    """Automatic initialization with sensible defaults."""
    global _INITIALIZED
    if not _INITIALIZED:
        try:
            # Try CUDA first, fall back to CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            initialize_device_dtype(device, "float64")
        except Exception as e:
            # If anything fails, use CPU + float64
            try:
                initialize_device_dtype("cpu", "float64")
            except:
                # Last resort - just set dtype
                torch.set_default_dtype(torch.float64)
                _INITIALIZED = True

# Auto-initialize when module is imported
_auto_initialize()
