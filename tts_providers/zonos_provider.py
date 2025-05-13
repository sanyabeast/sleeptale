"""
Zonos TTS Provider for SleepTeller.
This module implements the Zonos TTS provider interface.
"""

import re
import requests
from .base_provider import BaseTTSProvider


class ZonosTTSProvider(BaseTTSProvider):
    """Zonos TTS Provider implementation."""
    
    def __init__(self, config):
        """Initialize the Zonos TTS provider.
        
        Args:
            config: Dictionary containing Zonos-specific configuration
        """
        super().__init__(config)
        zonos_settings = config.get('zonos_settings', {})
        self.tts_server = zonos_settings.get('tts_server', 'http://localhost:5001/generate')
        self.voice_sample = zonos_settings.get('voice_sample')
    
    def preprocess_text(self, text):
        """Preprocess text for better compatibility with Zonos TTS.
        
        Args:
            text: The original text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Replace common problematic characters
        text = text.replace('"', '')  # Remove double quotes
        text = text.replace("'", '')  # Remove single quotes
        text = text.replace("...", ".")  # Replace ellipsis with period
        text = text.replace("—", "-")  # Replace em dash with hyphen
        text = text.replace("–", "-")  # Replace en dash with hyphen
        
        # Remove any remaining non-alphanumeric characters except basic punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def generate_voice_line(self, text, output_path, voice_options):
        """Generate voice line using Zonos TTS server.
        
        Args:
            text: The text to convert to speech
            output_path: Path where the audio file should be saved
            voice_options: Dictionary containing voice options
                - voice_sample: Path to voice sample file
                
        Returns:
            bool: True if successful, False otherwise
        """
        # Normalize the output path to use forward slashes
        normalized_output_path = str(output_path).replace('\\', '/')
        
        # Preprocess text for better TTS compatibility
        processed_text = self.preprocess_text(text)
        
        # Prepare request parameters
        params = {
            'text': processed_text,
            'path': normalized_output_path,
            'voice': voice_options.get('voice_sample'),
            'emotion': 'Neutral',  # Default emotion
        }
        
        try:
            print("🔄 Sending request to Zonos TTS server...")
            response = requests.get(self.tts_server, params=params)
            if response.status_code == 200:
                return True
            else:
                print(f"⚠️ Error generating voice line: {response.text}")
                return False
        except Exception as e:
            print(f"⚠️ Exception when calling Zonos TTS API: {str(e)}")
            return False
