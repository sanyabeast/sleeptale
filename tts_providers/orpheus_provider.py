"""
Orpheus TTS Provider for SleepTeller.
This module implements the Orpheus TTS provider interface.
"""

import re
import requests
import json
import os
from .base_provider import BaseTTSProvider


class OrpheusTTSProvider(BaseTTSProvider):
    """Orpheus TTS Provider implementation."""
    
    def __init__(self, config):
        """Initialize the Orpheus TTS provider.
        
        Args:
            config: Dictionary containing Orpheus-specific configuration
        """
        super().__init__(config)
        orpheus_settings = config.get('orpheus_settings', {})
        self.tts_server = orpheus_settings.get('tts_server', 'http://localhost:5005/v1/audio/speech')
        # Hardcoded values that don't need to be configurable
        self.model = 'orpheus'  # Always use the orpheus model
        self.response_format = 'wav'  # Always use WAV format
        
    def preprocess_text(self, text):
        """Preprocess text for better compatibility with Orpheus TTS.
        
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
        """Generate voice line using Orpheus TTS API.
        
        Args:
            text: The text to convert to speech
            output_path: Path where the audio file should be saved
            voice_options: Dictionary containing voice options
                - voice_preset: Voice preset name (e.g., 'jess')
                - speed: Speed factor (default: 1.0)
                
        Returns:
            bool: True if successful, False otherwise
        """
        # Preprocess text for better TTS compatibility
        processed_text = self.preprocess_text(text)
        
        # Get voice preset and speed from options
        voice_preset = voice_options.get('voice_preset', 'jess')
        speed = voice_options.get('speed', 1.0)
        
        # Prepare request payload
        payload = {
            'model': self.model,
            'input': processed_text,
            'voice': voice_preset,
            'response_format': self.response_format,
            'speed': speed
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        try:
            print("🔄 Sending request to Orpheus TTS server...")
            response = requests.post(self.tts_server, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                # Save the audio data to the output path
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"⚠️ Error generating voice line: {response.text}")
                return False
        except Exception as e:
            print(f"⚠️ Exception when calling Orpheus TTS API: {str(e)}")
            return False
