import os
import sys
import yaml
import glob
import time
import json
import random
import requests
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import argparse
import re
import shutil

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

def load_config():
    """Load configuration from config.yaml file."""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        log("Error: config.yaml not found in the root directory", "❌")
        sys.exit(1)
    except yaml.YAMLError as e:
        log(f"Error parsing config.yaml: {e}", "❌")
        sys.exit(1)

# Global variables
CONFIG = None
TTS_SERVER = None
VOICE_SAMPLE = None
SPEECH_RATE = None
STORIES_DIR = None
VOICE_LINES_DIR = None

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
    """Remove all files from the voice lines directory."""
    output_dir_path = os.path.join(PROJECT_DIR, "output", VOICE_LINES_DIR)
    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path)
        log("Cleaned voice lines directory", "🧹")
    ensure_dir_exists(output_dir_path)

def get_story_files():
    """Get all story YAML files."""
    story_files = []
    stories_path = os.path.join(PROJECT_DIR, "output", STORIES_DIR)
    
    # Ensure stories directory exists
    if not os.path.exists(stories_path):
        log(f"Stories directory not found: {stories_path}", "⚠️")
        return story_files
    
    for filename in os.listdir(stories_path):
        if filename.endswith('.yaml'):
            story_files.append(os.path.join(stories_path, filename))
    return story_files

def load_story(file_path):
    """Load story from YAML file."""
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

def generate_voice_line(text, output_path, selected_voice):
    """Generate voice line using TTS server."""
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
        'voice': selected_voice,
        'emotion': 'Neutral',  # Default emotion
    }
    
    try:
        log("Sending request to TTS server...", "🔄")
        response = requests.get(TTS_SERVER, params=params)
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

# Silence trimming functionality removed as requested

def process_story(story_file, force_regenerate=False, normalize_audio_setting=None, target_db=None):
    """Process a single story file and generate voice lines for all sentences."""
    # Extract story name from filename
    filename = os.path.basename(story_file)
    story_name = os.path.splitext(filename)[0]
    
    # Load story data
    story = load_story(story_file)
    
    log("\n" + "="*50)
    log(f"Generating voice lines for: {story['topic_title']}")
    log("="*50)
    
    # Calculate total sentences for progress tracking
    total_sentences = len(story['sentences'])
    log(f"Total sentences to process: {total_sentences}")
    
    # Get normalization settings from config
    if normalize_audio_setting is None:
        normalization_enabled = CONFIG.get("voice", {}).get("normalization", {}).get("enabled", False)
    else:
        normalization_enabled = normalize_audio_setting
    
    if target_db is None:
        target_db = CONFIG.get("voice", {}).get("normalization", {}).get("target_db", -20.0)
    
    if normalization_enabled:
        log(f"Audio normalization enabled (target: {target_db} dB)", "🔊")
    
    # Create story-specific output directory
    output_dir_path = os.path.join(PROJECT_DIR, "output", VOICE_LINES_DIR, story_name)
    ensure_dir_exists(output_dir_path)
    
    # Select a random voice for this story
    selected_voice = random.choice(VOICE_SAMPLE)
    log(f"Selected voice: {os.path.basename(selected_voice)}", "🎙️")
    
    # Process each sentence
    completed = 0
    skipped = 0
    for i, sentence in enumerate(story['sentences']):
        # Create output filename with zero-padded index
        output_filename = f"{i:04d}.wav"
        
        # Create absolute output path
        output_path = os.path.join(output_dir_path, output_filename)
        
        # Check if the file already exists and skip if not forcing regeneration
        if os.path.exists(output_path) and not force_regenerate:
            log(f"Skipping sentence {i:04d}/{total_sentences-1:04d}: File already exists", "⏩")
            skipped += 1
            continue
        
        # Show progress percentage
        progress_pct = (i / total_sentences) * 100
        log(f"Processing {i:04d}/{total_sentences-1:04d} ({progress_pct:.1f}%): \"{sentence}\"", "🔊")
        
        # Generate voice line
        if generate_voice_line(sentence, output_path, selected_voice):
            log(f"Generated: {output_filename}", "✅")
            completed += 1
            
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
        else:
            log(f"Failed to generate: {output_filename}", "❌")
    
    # Show summary for this story
    log(f"\nSummary for '{story['topic_title']}':", "📊")
    log(f"  - Total sentences: {total_sentences}", "📊")
    log(f"  - Generated: {completed}", "📊")
    log(f"  - Skipped: {skipped}", "📊")
    log(f"  - Failed: {total_sentences - completed - skipped}", "📊")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate voice lines for stories')
    parser.add_argument('-s', '--story', type=str,
                        help='Process only the specified story file (filename only, not full path)')
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
    """Main function to process all stories."""
    args = parse_arguments()
    
    # Load configuration
    global CONFIG, TTS_SERVER, VOICE_SAMPLE, SPEECH_RATE, STORIES_DIR, VOICE_LINES_DIR
    CONFIG = load_config()
    
    # Voice generation settings
    TTS_SERVER = CONFIG["voice"]["tts_server"]
    # Get list of voice samples
    VOICE_SAMPLE = CONFIG["voice"]["voice_sample"]
    if not isinstance(VOICE_SAMPLE, list):
        VOICE_SAMPLE = [VOICE_SAMPLE]  # Convert single voice to list for backward compatibility
    
    # No rate parameter used - using TTS server default
    
    # Directory settings
    STORIES_DIR = "stories"
    VOICE_LINES_DIR = "voice_lines"
    
    # Print startup message
    log(f"Starting voice line generation...", "🚀")
    
    # Clean output directory if requested
    if args.clean:
        clean_output_directory()
    else:
        ensure_dir_exists(os.path.join(PROJECT_DIR, "output", VOICE_LINES_DIR))
    
    # Get all story files
    story_files = get_story_files()
    log(f"Found {len(story_files)} story files", "📂")
    
    # Calculate total work to be done
    total_stories = len(story_files)
    if total_stories == 0:
        log("No stories found to process", "⚠️")
        return 0
    
    # Filter to a specific story if requested
    if args.story:
        story_files = [f for f in story_files if os.path.basename(f) == args.story]
        if not story_files:
            log(f"Error: Story file '{args.story}' not found", "❌")
            return 1
        log(f"Processing only story: {args.story}", "🔍")
    
    # Process each story
    for story_file in story_files:
        # Determine normalization setting
        normalize_audio_setting = None
        if args.normalize:
            normalize_audio_setting = True
        elif args.no_normalize:
            normalize_audio_setting = False
        
        # Process the story
        process_story(story_file, args.force, normalize_audio_setting, args.target_db)
    
    log("\nVoice line generation complete!", "🎉")
    return 0

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Process interrupted by user (Ctrl+C)", "⚠️")
        log("Exiting gracefully...", "🛑")
        sys.exit(130)  # Standard exit code for SIGINT
