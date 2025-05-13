"""
TTS Provider Factory for SleepTeller.
This module provides a factory function to create the appropriate TTS provider.
"""

from .zonos_provider import ZonosTTSProvider
from .orpheus_provider import OrpheusTTSProvider


def get_tts_provider(config):
    """Factory function to create the appropriate TTS provider.
    
    Args:
        config: Dictionary containing voice configuration
        
    Returns:
        BaseTTSProvider: An instance of the appropriate TTS provider
        
    Raises:
        ValueError: If the specified provider is not supported
    """
    provider_name = config.get('provider', 'zonos').lower()
    
    if provider_name == 'zonos':
        return ZonosTTSProvider(config)
    elif provider_name == 'orpheus':
        return OrpheusTTSProvider(config)
    else:
        raise ValueError(f"Unsupported TTS provider: {provider_name}")
