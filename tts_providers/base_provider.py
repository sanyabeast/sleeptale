"""
Base TTS Provider Interface for SleepTeller.
This module defines the interface that all TTS providers must implement.
"""

from abc import ABC, abstractmethod


class BaseTTSProvider(ABC):
    """Base class for all TTS providers."""
    
    def __init__(self, config):
        """Initialize the TTS provider with configuration.
        
        Args:
            config: Dictionary containing provider-specific configuration
        """
        self.config = config
    
    @abstractmethod
    def generate_voice_line(self, text, output_path, voice_options):
        """Generate a voice line using the TTS provider.
        
        Args:
            text: The text to convert to speech
            output_path: Path where the audio file should be saved
            voice_options: Provider-specific voice options
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def preprocess_text(self, text):
        """Preprocess text for better compatibility with the TTS provider.
        
        Args:
            text: The original text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        pass
