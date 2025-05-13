"""
Utility functions for TTS providers.
"""

def normalize_path(path):
    """Normalize path to use forward slashes consistently."""
    return str(path).replace('\\', '/')
