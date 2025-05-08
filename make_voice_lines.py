import os
import yaml
import requests
import re
import shutil
import argparse
import random
import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# Get the absolute path of the project directory
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

# ─────────────────────────────────────────────────────
# 🎲 CONFIGURATION
# ─────────────────────────────────────────────────────
def log(message, emoji=None):
    """Standardized logging function with consistent emoji spacing.
    
    Args:
        message: The message to log
        emoji: Optional emoji to prefix the message
    """
    if emoji:
        # Ensure there's a space after the emoji
        print(f"{emoji} {message}")
    else:
        print(message)

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        log("Error: No config file specified.", "❌")
        log("Hint: Use -c or --config to specify a config file. Example: -c configs/sample.yaml", "💡")
        sys.exit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Extract project name from config filename if not specified
        if 'project_name' not in config:
            # Get the filename without extension
            config_filename = os.path.basename(config_path)
            config_name = os.path.splitext(config_filename)[0]
            config['project_name'] = config_name
            log(f"Using config filename '{config_name}' as project name", "ℹ️")
        
        return config
    except FileNotFoundError:
        log(f"Error: Config file not found: {config_path}", "❌")
        log(f"Hint: Make sure the config file exists. Example: configs/sample.yaml", "💡")
        sys.exit(1)
    except yaml.YAMLError as e:
        log(f"Error parsing config file: {e}", "❌")
        sys.exit(1)

# Global variables
CONFIG = None
ZONOS_TTS_SERVER = None
VOICE_SAMPLES = None
SPEECH_RATE = None
SCENARIOS_DIR = None
OUTPUT_DIR = None

# ─────────────────────────────────────────────────────
# 🐍 UTILITIES
# ─────────────────────────────────────────────────────
def to_snake_case(text: str) -> str:
    """Convert text to snake_case."""
    text = re.sub(r'[^\w\s]', '', text).lower()
    return re.sub(r'\s+', '_', text)

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def clean_output_directory():
    """Remove all files from the output directory."""
    project_name = CONFIG.get("project_name", "DeepVideo2")
    output_dir_path = os.path.join(PROJECT_DIR, "output", project_name, OUTPUT_DIR)
    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path)
        log("Cleaned output directory", "🧹")
    ensure_dir_exists(output_dir_path)

def get_scenario_files():
    """Get all scenario YAML files."""
    scenario_files = []
    project_name = CONFIG.get("project_name", "DeepVideo2")
    scenarios_path = os.path.join(PROJECT_DIR, "output", project_name, SCENARIOS_DIR)
    
    # Ensure scenarios directory exists
    if not os.path.exists(scenarios_path):
        log(f"Scenarios directory not found: {scenarios_path}", "⚠️")
        return scenario_files
    
    for filename in os.listdir(scenarios_path):
        if filename.endswith('.yaml'):
            scenario_files.append(os.path.join(scenarios_path, filename))
    return scenario_files

def load_scenario(file_path):
    """Load scenario from YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def normalize_path(path):
    """Normalize path to use forward slashes consistently."""
    return str(Path(path)).replace('\\', '/')

def preprocess_text_for_tts(text):
    """Preprocess text to make it more compatible with Zonos TTS server.
    
    Args:
        text: The original text to preprocess
        
    Returns:
        Preprocessed text with problematic characters removed/replaced
    """
    if not text:
        return text
    
    # 1. Remove all double quotes
    text = text.replace('"', "")
    
    # 2. Replace three-dots (ellipsis) with a single dot
    text = text.replace("...", ".").replace(". . .", ".")
    
    # 3. Remove the word "ugh" (case insensitive)
    text = re.sub(r'\bugh\b', '', text, flags=re.IGNORECASE)
    
    # Remove any double spaces created by the replacements
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def generate_voice_line(text, output_path, emotion, voice_sample):
    """Generate voice line using Zonos TTS server."""
    # Normalize the output path to use forward slashes
    normalized_output_path = normalize_path(output_path)
    
    # Preprocess text for better TTS compatibility
    original_text = text
    processed_text = preprocess_text_for_tts(text)
    
    # Log if text was modified
    if processed_text != original_text:
        log(f"Preprocessed text: \"{original_text}\" → \"{processed_text}\"", "🔄")
    
    params = {
        'text': processed_text,
        'path': normalized_output_path,
        'voice': voice_sample,
        'emotion': emotion.capitalize(),  # Capitalize emotion for the API
        'rate': SPEECH_RATE
    }
    
    try:
        log("Sending request to TTS server...", "🔄")
        response = requests.get(ZONOS_TTS_SERVER, params=params)
        if response.status_code == 200:
            return True
        else:
            log(f"Error generating voice line: {response.text}", "⚠️")
            return False
    except Exception as e:
        log(f"Exception when calling TTS API: {str(e)}", "⚠️")
        return False

def normalize_audio(input_path, output_path=None, target_db=-20.0):
    """
    Normalize audio file to a target dB level.
    
    Args:
        input_path: Path to the input audio file
        output_path: Path to save the normalized audio (if None, overwrites input)
        target_db: Target dB level (default: -20.0)
    
    Returns:
        Tuple of (success, original_duration, new_duration)
    """
    try:
        # Load the audio file
        y, sr = librosa.load(input_path, sr=None)
        
        # Get the original duration for logging
        original_duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate current RMS energy
        rms = np.sqrt(np.mean(y**2))
        current_db = 20 * np.log10(rms)
        
        # Calculate the gain needed
        gain = 10 ** ((target_db - current_db) / 20)
        
        # Apply gain to normalize
        y_normalized = y * gain
        
        # Ensure we don't clip
        max_val = np.max(np.abs(y_normalized))
        if max_val > 1.0:
            y_normalized = y_normalized / max_val * 0.99
        
        # Determine output path
        if output_path is None:
            output_path = input_path
        
        # Export the normalized audio
        sf.write(output_path, y_normalized, sr)
        
        # Calculate new duration (should be the same)
        new_duration = librosa.get_duration(y=y_normalized, sr=sr)
        
        return True, original_duration, new_duration
    except Exception as e:
        log(f"Error normalizing {input_path}: {str(e)}", "⚠️")
        return False, 0, 0

def trim_silence_from_end(input_path, output_path=None, max_silence_sec=1.0, threshold_db=-50):
    """
    Trim silence from the end of an audio file, keeping up to max_silence_sec seconds of silence.
    
    Args:
        input_path: Path to the input audio file
        output_path: Path to save the trimmed audio (if None, overwrites input)
        max_silence_sec: Maximum silence to keep at the end in seconds (default: 1.0)
        threshold_db: Threshold in dB below which audio is considered silence (default: -50)
    
    Returns:
        Tuple of (success, original_duration, new_duration)
    """
    try:
        # Load the audio file
        y, sr = librosa.load(input_path, sr=None)
        
        # Get the original duration for logging
        original_duration = librosa.get_duration(y=y, sr=sr)
        
        # Convert threshold from dB to amplitude
        threshold_amp = 10 ** (threshold_db / 20)
        
        # Find the last non-silent sample
        # Start from the end and move backwards
        last_idx = len(y) - 1
        while last_idx >= 0 and abs(y[last_idx]) < threshold_amp:
            last_idx -= 1
        
        # If the entire file is silent, keep a small portion
        if last_idx < 0:
            log(f"Audio file appears to be entirely silent: {input_path}", "⚠️")
            # Keep just a small portion of silence
            new_y = y[:int(sr * 0.5)]  # 0.5 seconds
        else:
            # Add the desired amount of silence after the last non-silent sample
            silence_samples = int(sr * max_silence_sec)
            end_idx = min(last_idx + silence_samples, len(y))
            new_y = y[:end_idx]
        
        # Determine output path
        if output_path is None:
            output_path = input_path
        
        # Export the trimmed audio
        sf.write(output_path, new_y, sr)
        
        # Calculate new duration
        new_duration = librosa.get_duration(y=new_y, sr=sr)
        
        return True, original_duration, new_duration
    except Exception as e:
        log(f"Error trimming silence from {input_path}: {str(e)}", "⚠️")
        return False, 0, 0

def process_scenario(scenario_file, force_regenerate=False, normalize_audio_setting=None, target_db=None):
    """Process a single scenario file and generate voice lines for all slides."""
    # Extract scenario name from filename
    filename = os.path.basename(scenario_file)
    scenario_name = os.path.splitext(filename)[0]
    
    # Load scenario data
    scenario = load_scenario(scenario_file)
    
    log("\n" + "="*50)
    log(f"Generating voice lines for: {scenario['topic']}")
    log("="*50)
    
    # Select a random voice sample for this scenario
    voice_sample = random.choice(VOICE_SAMPLES)
    voice_name = os.path.basename(voice_sample)
    log(f"Selected voice: {voice_name}", "🎙️")
    
    # Get normalization settings from config
    if normalize_audio_setting is None:
        normalization_enabled = CONFIG.get("voice", {}).get("normalization", {}).get("enabled", False)
    else:
        normalization_enabled = normalize_audio_setting
    
    if target_db is None:
        target_db = CONFIG.get("voice", {}).get("normalization", {}).get("target_db", -20.0)
    
    if normalization_enabled:
        log(f"Audio normalization enabled (target: {target_db} dB)", "🔊")
    
    # Get silence trimming settings from config
    silence_trimming_enabled = CONFIG.get("voice", {}).get("silence_trimming", {}).get("enabled", True)
    max_silence_sec = CONFIG.get("voice", {}).get("silence_trimming", {}).get("max_silence_sec", 1.0)
    threshold_db = CONFIG.get("voice", {}).get("silence_trimming", {}).get("threshold_db", -50)
    
    if silence_trimming_enabled:
        log(f"Silence trimming enabled (max: {max_silence_sec}s, threshold: {threshold_db} dB)", "✂️")
    
    # Create project-specific output directory
    project_name = CONFIG.get("project_name", "DeepVideo2")
    output_dir_path = os.path.join(PROJECT_DIR, "output", project_name, OUTPUT_DIR)
    ensure_dir_exists(output_dir_path)
    
    # Process each slide
    for i, slide in enumerate(scenario['slides']):
        # Create output filename
        slide_id = f"slide_{i+1:02d}"
        output_filename = f"{scenario_name}_{slide_id}.wav"
        
        # Create absolute output path
        output_path = os.path.join(output_dir_path, output_filename)
        
        # Check if the file already exists and skip if not forcing regeneration
        if os.path.exists(output_path) and not force_regenerate:
            log(f"Skipping slide {i+1}: File already exists", "⏩")
            continue
        
        # Get slide text and emotion
        text = slide['text']
        emotion = slide['emotion']
        
        log(f"Slide {i+1}: \"{text}\" - {emotion}", "🔊")
        
        # Generate voice line with the selected voice sample
        if generate_voice_line(text, output_path, emotion, voice_sample):
            log(f"Generated: {output_filename}", "✅")
            
            # Normalize audio if enabled
            if normalization_enabled and os.path.exists(output_path):
                log(f"Normalizing audio to {target_db} dB...", "🔄")
                success, original_duration, new_duration = normalize_audio(
                    output_path, None, target_db
                )
                if success:
                    log(f"Normalized: {output_filename} ({original_duration:.2f}s → {new_duration:.2f}s)", "✅")
                else:
                    log(f"Failed to normalize: {output_filename}", "⚠️")
                
                # Trim silence from the end if enabled
                if silence_trimming_enabled and os.path.exists(output_path):
                    log(f"Trimming silence from the end (max: {max_silence_sec}s)...", "✂️")
                    success, original_duration, new_duration = trim_silence_from_end(
                        output_path, None, max_silence_sec, threshold_db
                    )
                    if success:
                        log(f"Trimmed silence: {output_filename} ({original_duration:.2f}s → {new_duration:.2f}s)", "✅")
                    else:
                        log(f"Failed to trim silence: {output_filename}", "⚠️")
        else:
            log(f"Failed to generate: {output_filename}", "❌")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate voice lines for scenarios')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to the configuration file')
    parser.add_argument('--clean', action='store_true', 
                        help='Remove all existing voice lines before generation')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration of voice lines even if they already exist')
    parser.add_argument('--normalize', action='store_true',
                        help='Force audio normalization even if disabled in config')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable audio normalization even if enabled in config')
    parser.add_argument('--target-db', type=float,
                        help='Target dB level for audio normalization (overrides config)')
    return parser.parse_args()

def main():
    """Main function to process all scenarios."""
    args = parse_arguments()
    
    # Load configuration from specified file
    global CONFIG, ZONOS_TTS_SERVER, VOICE_SAMPLES, SPEECH_RATE, SCENARIOS_DIR, OUTPUT_DIR
    CONFIG = load_config(args.config)
    
    # Voice generation settings
    ZONOS_TTS_SERVER = CONFIG["voice"]["zonos_tts_server"]
    VOICE_SAMPLES = CONFIG["voice"]["voice_samples"]
    SPEECH_RATE = CONFIG["voice"]["speech_rate"]
    
    # Directory settings
    SCENARIOS_DIR = CONFIG["directories"]["scenarios"]
    OUTPUT_DIR = CONFIG["directories"]["voice_lines"]
    
    # Get project name
    project_name = CONFIG.get("project_name")
    
    # Print startup message
    log(f"Starting {project_name} voice line generation...", "🚀")
    
    # Clean output directory if requested
    output_dir_path = os.path.join(PROJECT_DIR, "output", project_name, OUTPUT_DIR)
    if args.clean:
        clean_output_directory()
    else:
        ensure_dir_exists(output_dir_path)
    
    # Get all scenario files
    scenario_files = get_scenario_files()
    log(f"Found {len(scenario_files)} scenario files", "📂")
    
    # Process each scenario
    for scenario_file in scenario_files:
        normalize_audio_setting = None
        target_db = None
        if args.normalize:
            normalize_audio_setting = True
        elif args.no_normalize:
            normalize_audio_setting = False
        if args.target_db:
            target_db = args.target_db
        process_scenario(scenario_file, args.force, normalize_audio_setting, target_db)
    
    log("Voice line generation complete!", "🎉")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Process interrupted by user (Ctrl+C)", "⚠️")
        log("Exiting gracefully...", "🛑")
        sys.exit(130)  # Standard exit code for SIGINT
