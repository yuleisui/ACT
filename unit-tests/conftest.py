import sys
from pathlib import Path

# Insert project root so top-level packages (act, input_parser, etc.) are importable
# Add both project root and act directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACT_DIR = PROJECT_ROOT / 'act'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ACT_DIR))