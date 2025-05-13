"""
StyleTTS2 TTS Provider for SleepTeller.
This module implements the StyleTTS2 TTS provider interface using the Gradio client.
"""

import re
import os
import time
import shutil
from gradio_client import Client
from .base_provider import BaseTTSProvider


class StyleTTS2Provider(BaseTTSProvider):
    """Provider for StyleTTS2 API.
    
    This provider interacts with the StyleTTS2 API to generate voice lines using the Gradio client.
    """
    
    def __init__(self, config):
        """Initialize the StyleTTS2 provider with configuration.
        
        Args:
            config: Configuration dictionary containing StyleTTS2 settings
        """
        self.config = config
        self.styletts2_config = config.get('styletts2_settings', {})
        self.base_url = self.styletts2_config.get('base_url', 'http://127.0.0.1:7860')
        # Remove /gradio_api/call suffix if present
        if self.base_url.endswith('/gradio_api/call'):
            self.base_url = self.base_url.replace('/gradio_api/call', '')
        
        # Get voice presets and speed from config
        self.voice_presets = self.styletts2_config.get('voice_presets', ['Richard_Male_EN_US'])
        if isinstance(self.voice_presets, str):
            self.voice_presets = [self.voice_presets]
        self.speed = self.styletts2_config.get('speed', 50)
        
        # Initialize Gradio client
        try:
            self.client = Client(self.base_url)
            print(f"✅ Connected to StyleTTS2 API at {self.base_url}")
        except Exception as e:
            print(f"⚠️ Failed to connect to StyleTTS2 API: {str(e)}")
            self.client = None
    
    def preprocess_text(self, text):
        """Preprocess text for better compatibility with StyleTTS2.
        
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
    
    def generate_voice_line(self, text, output_path, voice_options=None):
        """Generate a voice line using StyleTTS2.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the generated audio
            voice_options: Optional voice customization parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if client is available
        if not self.client:
            print("⚠️ StyleTTS2 client not initialized. Check connection to API server.")
            return False
            
        # Use voice options if provided, otherwise select randomly from presets
        if voice_options and 'voice' in voice_options:
            voice = voice_options['voice']
        else:
            # Select a random voice from the presets
            import random
            voice = random.choice(self.voice_presets)
            
        # Get speed from options or config
        speed = voice_options.get('speed', self.speed) if voice_options else self.speed
        
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Generate voice line using the basic method
        return self._generate_basic(text, output_path, voice, speed)
    
    def _generate_basic(self, text, output_path, voice, speed):
        """Generate voice using the basic TTS endpoint.
        
        Args:
            text: The text to convert to speech
            output_path: Path where the audio file should be saved
            voice: Voice preset name
            speed: Speed percentage
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"🔄 Calling StyleTTS2 API with text: '{text[:30]}...', voice: {voice}, speed: {speed}")
            
            # Call the Gradio API using the client
            result = self.client.predict(
                text,
                voice,
                speed,
                api_name="/on_generate_tts"
            )
            
            # The result is a tuple with (filepath, status_message)
            audio_file, status = result
            
            print(f"✅ StyleTTS2 API response: {status}")
            
            # Copy the generated audio file to the output path
            shutil.copy2(audio_file, output_path)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Exception when using StyleTTS2: {str(e)}")
            return False
    
    # Studio method removed as we're using only basic voice generation with multiple voice presets
