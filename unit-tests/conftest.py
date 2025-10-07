import sys
from pathlib import Path

# Insert project root so top-level packages (verifier, input_parser, etc.) are importable
# Add both project root and verifier directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERIFIER_DIR = PROJECT_ROOT / 'verifier'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VERIFIER_DIR))