"""
zonation-image analysis - Python utilities for image analyses of zonation patterns.
"""
from pathlib import Path

__author__ = "Matthias König"
__version__ = "0.0.2"
program_name = "zonation-image-analysis"

BASE_PATH = Path(__file__).parent.parent.parent
RESOURCES_PATH = BASE_PATH / "src" / "zia" / "resources"
