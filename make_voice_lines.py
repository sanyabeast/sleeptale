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
from tqdm import tqdm
import datetime

# Import TTS provider factory
from tts_providers import get_tts_provider

# Import pyrubberband for high-quality time-stretching
try:
    import pyrubberband as pyrb
    import os
    import subprocess
    import sys
    
    # Print debugging information
    print("\n=== Rubberband Detection Debug Info ===")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # On Windows, check for different possible executable names
    RUBBERBAND_AVAILABLE = False
    possible_executables = ["rubberband.exe", "rubberband-cli.exe", "rubberband", "rubberband-r3.exe"]
    
    # Get the PATH directories
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    print(f"Number of PATH directories: {len(path_dirs)}")
    
    # Check if any of the possible executables exist in PATH
    for path_dir in path_dirs:
        for exe in possible_executables:
            exe_path = os.path.join(path_dir, exe)
            if os.path.isfile(exe_path):
                print(f"✅ Found Rubberband executable: {exe_path}")
                # Monkey patch pyrubberband to use the correct executable name
                pyrb.pyrb._RUBBERBAND_UTIL = exe
                RUBBERBAND_AVAILABLE = True
                break
        if RUBBERBAND_AVAILABLE:
            break
    
    # If not found in PATH, try looking in common installation directories
    if not RUBBERBAND_AVAILABLE:
        common_dirs = [
            "C:\\Program Files\\Rubberband",
            "C:\\Program Files (x86)\\Rubberband",
            os.path.join(os.environ.get("USERPROFILE", ""), "Rubberband"),
            os.path.join(os.environ.get("USERPROFILE", ""), "Downloads", "Rubberband"),
            os.path.dirname(os.path.abspath(__file__))  # Current script directory
        ]
        
        for common_dir in common_dirs:
            if os.path.exists(common_dir):
                print(f"Checking directory: {common_dir}")
                for exe in possible_executables:
                    exe_path = os.path.join(common_dir, exe)
                    if os.path.isfile(exe_path):
                        print(f"✅ Found Rubberband executable: {exe_path}")
                        # Monkey patch pyrubberband to use the correct executable name with full path
                        pyrb.pyrb._RUBBERBAND_UTIL = exe_path
                        RUBBERBAND_AVAILABLE = True
                        break
            if RUBBERBAND_AVAILABLE:
                break
                
    # Try to load Rubberband path from config file
    try:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            rubberband_path = config.get('voice', {}).get('rubberband', {}).get('path')
            if rubberband_path and os.path.isfile(rubberband_path):
                print(f"✅ Found Rubberband executable from config: {rubberband_path}")
                # Monkey patch pyrubberband to use the specified executable path
                pyrb.pyrb._RUBBERBAND_UTIL = rubberband_path
                RUBBERBAND_AVAILABLE = True
    except Exception as e:
        print(f"Error loading Rubberband path from config: {str(e)}")
    
    # Manual path specification - if you know where rubberband.exe is located
    if not RUBBERBAND_AVAILABLE:
        # Try to find it in the current directory
        for exe in possible_executables:
            if os.path.isfile(exe):
                print(f"✅ Found Rubberband executable in current directory: {os.path.abspath(exe)}")
                # Use full path to be safe
                pyrb.pyrb._RUBBERBAND_UTIL = os.path.abspath(exe)
                RUBBERBAND_AVAILABLE = True
                break
    
    if RUBBERBAND_AVAILABLE:
        print(f"Using Rubberband executable: {pyrb.pyrb._RUBBERBAND_UTIL}")
        # Test if it actually works
        try:
            import numpy as np
            test_audio = np.zeros(1000)
            test_result = pyrb.time_stretch(test_audio, 44100, 1.0)
            print("✅ Rubberband test successful!")
        except Exception as e:
            print(f"⚠️ Rubberband test failed: {str(e)}")
            RUBBERBAND_AVAILABLE = False
    else:
        print("⚠️ Rubberband executable not found, falling back to librosa for time-stretching")
    
    print("=== End Debug Info ===\n")
        
except ImportError as e:
    RUBBERBAND_AVAILABLE = False
    print(f"⚠️ pyrubberband package not found: {str(e)}, falling back to librosa for time-stretching")

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
TTS_PROVIDER = None
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
    """Preprocess text to make it more compatible with TTS providers.
    
    Args:
        text: The original text to preprocess
        
    Returns:
        Preprocessed text with problematic characters removed/replaced
    """
    # This function is now delegated to the TTS provider
    # but we keep it for backward compatibility
    return TTS_PROVIDER.preprocess_text(text)

def generate_voice_line(text, output_path, voice_options):
    """Generate voice line using the configured TTS provider.
    
    Args:
        text: The text to convert to speech
        output_path: Path where the audio file should be saved
        voice_options: Provider-specific voice options
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Preprocess text for better TTS compatibility
    original_text = text
    processed_text = preprocess_text_for_tts(text)
    
    # Log if text was modified
    if processed_text != original_text:
        log(f"Preprocessed text: \"{original_text}\" → \"{processed_text}\"", "🔄")
    
    # Use the provider to generate the voice line
    return TTS_PROVIDER.generate_voice_line(processed_text, output_path, voice_options)

def use_rubberband_direct(input_path, output_path, tempo_factor):
    """
    Use Rubberband directly via subprocess to change tempo.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        tempo_factor: Factor to change tempo (< 1.0 slows down, > 1.0 speeds up)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get Rubberband path from config
        rubberband_path = CONFIG.get('voice', {}).get('rubberband', {}).get('path')
        if not rubberband_path or not os.path.isfile(rubberband_path):
            log("Rubberband executable not found in config", "⚠️")
            return False
        
        # Calculate time ratio (inverse of tempo factor)
        time_ratio = 1.0 / tempo_factor
        
        # Construct command
        # --time <ratio>: time stretch ratio (e.g. 2.0 = double duration)
        # --tempo <factor>: tempo change factor (e.g. 0.5 = half speed)
        # --pitch-hq: high quality pitch preservation
        # --formant: preserve formants for more natural voice
        # -c <n>: crispness (0-6, higher = better quality)
        cmd = [
            rubberband_path,
            "--tempo", str(tempo_factor),
            "--pitch-hq",
            "--formant",
            "-c", "6",
            input_path,
            output_path
        ]
        
        log(f"Running Rubberband command: {' '.join(cmd)}", "🔄")
        
        # Run command
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            log("Rubberband processing successful", "✅")
            return True
        else:
            log(f"Rubberband error: {result.stderr}", "⚠️")
            return False
    except Exception as e:
        log(f"Error using Rubberband directly: {str(e)}", "⚠️")
        return False

def change_tempo(audio_data, sr, tempo_factor=0.9):
    """
    Change the tempo of audio without affecting pitch.
    Uses Rubberband for high-quality results if available, otherwise falls back to librosa.
    
    Args:
        audio_data: Audio data as numpy array
        sr: Sample rate
        tempo_factor: Factor to change tempo (< 1.0 slows down, > 1.0 speeds up)
        
    Returns:
        Audio data with modified tempo
    """
    try:
        # Make sure tempo_factor is not 0 or negative
        if tempo_factor <= 0:
            log(f"Invalid tempo factor: {tempo_factor}, using 0.9 instead", "⚠️")
            tempo_factor = 0.9
            
        log(f"Applying tempo change with factor: {tempo_factor} (slower)", "🔄")
        
        # Try using Rubberband directly via subprocess
        # This requires saving to a temporary file and reading back
        try:
            # Get Rubberband path from config
            rubberband_path = CONFIG.get('voice', {}).get('rubberband', {}).get('path')
            if rubberband_path and os.path.isfile(rubberband_path):
                # Create temporary files
                import tempfile
                temp_dir = tempfile.gettempdir()
                temp_input = os.path.join(temp_dir, "temp_input.wav")
                temp_output = os.path.join(temp_dir, "temp_output.wav")
                
                # Write audio to temporary file
                sf.write(temp_input, audio_data, sr)
                
                # Process with Rubberband
                if use_rubberband_direct(temp_input, temp_output, tempo_factor):
                    # Read processed audio
                    processed_audio, processed_sr = librosa.load(temp_output, sr=None)
                    
                    # Clean up temporary files
                    try:
                        os.remove(temp_input)
                        os.remove(temp_output)
                    except:
                        pass
                    
                    log(f"Successfully processed with Rubberband directly", "✅")
                    return processed_audio, processed_sr
        except Exception as e:
            log(f"Error with direct Rubberband approach: {str(e)}", "⚠️")
        
        # Fall back to librosa
        log(f"Using librosa for time-stretching with rate={tempo_factor}", "🎵")
        
        # For librosa, we need to use tempo_factor directly
        # In librosa 0.9.0+, the parameter is 'rate' and it's directly what we want for slowing down
        # tempo_factor=0.5 means play at half speed
        stretched_audio = librosa.effects.time_stretch(audio_data, rate=tempo_factor)
        return stretched_audio, sr
    except Exception as e:
        log(f"Error changing tempo: {str(e)}", "⚠️")
        # Return original audio if there's an error
        return audio_data, sr

def add_echo_effect(audio_data, sr, delay=0.3, decay=0.5):
    """
    Add echo effect to audio data.
    
    Args:
        audio_data: Audio data as numpy array
        sr: Sample rate
        delay: Delay time in seconds
        decay: Echo volume decay (0-1)
        
    Returns:
        Audio data with echo effect
    """
    try:
        # Convert delay from seconds to samples
        delay_samples = int(delay * sr)
        
        # Create delayed version of the audio
        delayed_audio = np.zeros_like(audio_data)
        if delay_samples < len(audio_data):
            delayed_audio[delay_samples:] = audio_data[:-delay_samples] * decay
        
        # Mix original and delayed audio
        echo_audio = audio_data + delayed_audio
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(echo_audio))
        if max_val > 1.0:
            echo_audio = echo_audio / max_val * 0.99
            
        return echo_audio
    except Exception as e:
        log(f"Error adding echo effect: {str(e)}", "⚠️")
        return audio_data

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

def apply_audio_effects(input_path, output_path=None, tempo_settings=None, echo_settings=None):
    """
    Apply audio effects (tempo change and echo) to an audio file.
    
    Args:
        input_path: Path to the input audio file
        output_path: Path to save the processed audio (if None, overwrites input)
        tempo_settings: Dictionary with tempo effect settings
        echo_settings: Dictionary with echo effect settings
    
    Returns:
        Tuple of (success, original_duration, new_duration)
    """
    try:
        # Load the audio file
        y, sr = librosa.load(input_path, sr=None)
        
        # Get the original duration for logging
        original_duration = librosa.get_duration(y=y, sr=sr)
        modified_audio = y
        modified_sr = sr
        
        # Apply tempo change if enabled
        if tempo_settings and tempo_settings.get('enabled', False):
            tempo_factor = tempo_settings.get('factor', 0.9)
            log(f"Changing tempo (factor: {tempo_factor})...", "🔄")
            
            # Validate tempo factor
            if tempo_factor <= 0 or tempo_factor > 2.0:
                log(f"Warning: Extreme tempo factor {tempo_factor} may produce unexpected results", "⚠️")
                
            # Apply tempo change
            modified_audio, modified_sr = change_tempo(modified_audio, sr, tempo_factor)
            
            # Verify the duration change
            expected_duration = original_duration / tempo_factor
            actual_duration = librosa.get_duration(y=modified_audio, sr=modified_sr)
            log(f"Original duration: {original_duration:.2f}s, Expected new duration: {expected_duration:.2f}s, Actual: {actual_duration:.2f}s", "📊")
        
        # Apply echo effect if enabled
        if echo_settings and echo_settings.get('enabled', False):
            delay = echo_settings.get('delay', 0.3)
            decay = echo_settings.get('decay', 0.5)
            log(f"Adding echo effect (delay: {delay}s, decay: {decay})...", "🔄")
            modified_audio = add_echo_effect(modified_audio, modified_sr, delay, decay)
        
        # Determine output path
        if output_path is None:
            output_path = input_path
        
        # Export the processed audio
        sf.write(output_path, modified_audio, modified_sr)
        
        # Calculate new duration
        new_duration = librosa.get_duration(y=modified_audio, sr=modified_sr)
        
        return True, original_duration, new_duration
    except Exception as e:
        log(f"Error applying audio effects to {input_path}: {str(e)}", "⚠️")
        return False, 0, 0

# Silence trimming functionality removed as requested

def process_story(story_file, force_regenerate=False, normalize_audio_setting=None, target_db=None, voice_index=None, 
              tempo_setting=None, tempo_factor=None, echo_setting=None, echo_delay=None, echo_decay=None):
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
    
    # Get postprocessing settings from config
    postprocessing = CONFIG.get("voice", {}).get("postprocessing", {})
    
    # Get normalization settings from config or command line arguments
    if normalize_audio_setting is None:
        normalize_audio_setting = postprocessing.get("normalization", {}).get("enabled", True)
    if target_db is None:
        target_db = postprocessing.get("normalization", {}).get("target_db", -20.0)
        
    # Get tempo settings from config or command line arguments
    if tempo_setting is None:
        tempo_setting = postprocessing.get("tempo", {}).get("enabled", False)
    if tempo_factor is None:
        tempo_factor = postprocessing.get("tempo", {}).get("factor", 0.9)
        
    # Get echo settings from config or command line arguments
    if echo_setting is None:
        echo_setting = postprocessing.get("echo", {}).get("enabled", False)
    if echo_delay is None:
        echo_delay = postprocessing.get("echo", {}).get("delay", 0.3)
    if echo_decay is None:
        echo_decay = postprocessing.get("echo", {}).get("decay", 0.5)
    
    # Log audio processing settings
    if normalize_audio_setting:
        log(f"Audio normalization enabled (target: {target_db} dB)", "🔊")
    
    if tempo_setting:
        log(f"Tempo manipulation enabled (factor: {tempo_factor})", "🔊")
    
    if echo_setting:
        log(f"Echo effect enabled (delay: {echo_delay}s, decay: {echo_decay})", "🔊")
    
    # Create story-specific output directory
    output_dir_path = os.path.join(PROJECT_DIR, "output", VOICE_LINES_DIR, story_name)
    ensure_dir_exists(output_dir_path)
    
    # Prepare voice options based on the provider
    voice_options = {}
    provider_name = CONFIG.get('voice', {}).get('provider', 'zonos').lower()
    
    if provider_name == 'zonos':
        # For Zonos, we need to handle voice samples
        zonos_settings = CONFIG.get('voice', {}).get('zonos_settings', {})
        voice_sample = zonos_settings.get('voice_sample', None)
        
        if isinstance(voice_sample, list):
            if voice_index is not None and 0 <= voice_index < len(voice_sample):
                selected_voice = voice_sample[voice_index]
                log(f"Using specified voice {voice_index}: {os.path.basename(selected_voice)}", "🎙️")
            else:
                # Use a random voice if index is out of range or not specified
                selected_voice = random.choice(voice_sample)
                log(f"Selected random voice: {os.path.basename(selected_voice)}", "🎙️")
        else:
            selected_voice = voice_sample
            log(f"Using voice: {os.path.basename(selected_voice)}", "🎙️")
            
        voice_options['voice_sample'] = selected_voice
            
    elif provider_name == 'orpheus':
        # For Orpheus, we use voice presets
        orpheus_settings = CONFIG.get('voice', {}).get('orpheus_settings', {})
        voice_presets = orpheus_settings.get('voice_presets', ['jess'])
        
        if voice_index is not None and 0 <= voice_index < len(voice_presets):
            selected_preset = voice_presets[voice_index]
            log(f"Using specified voice preset {voice_index}: {selected_preset}", "🎤")
        else:
            # Use a random preset if index is out of range or not specified
            selected_preset = random.choice(voice_presets)
            log(f"Selected random voice preset: {selected_preset}", "🎤")
            
        voice_options['voice_preset'] = selected_preset
        voice_options['speed'] = orpheus_settings.get('speed', 1.0)
        
    elif provider_name == 'styletts2':
        # For StyleTTS2, we use voice presets
        styletts2_settings = CONFIG.get('voice', {}).get('styletts2_settings', {})
        voice_presets = styletts2_settings.get('voice_presets', ['Richard_Male_EN_US'])
        
        if voice_index is not None and 0 <= voice_index < len(voice_presets):
            selected_voice = voice_presets[voice_index]
            log(f"Using specified StyleTTS2 voice {voice_index}: {selected_voice}", "🎤")
        else:
            # Use a random voice if index is out of range or not specified
            selected_voice = random.choice(voice_presets)
            log(f"Selected random StyleTTS2 voice: {selected_voice}", "🎤")
            
        voice_options['voice'] = selected_voice
        voice_options['speed'] = styletts2_settings.get('speed', 50)
    
    # Get provider-specific settings
    provider_name = CONFIG.get('voice', {}).get('provider', 'zonos').lower()
    
    # Get number of sentences per voice line from provider-specific config
    if provider_name == 'zonos':
        zonos_settings = CONFIG.get('voice', {}).get('zonos_settings', {})
        sentences_per_line = zonos_settings.get('sentences_per_voice_line', 1)
    elif provider_name == 'orpheus':
        orpheus_settings = CONFIG.get('voice', {}).get('orpheus_settings', {})
        sentences_per_line = orpheus_settings.get('sentences_per_voice_line', 1)
    elif provider_name == 'styletts2':
        styletts2_settings = CONFIG.get('voice', {}).get('styletts2_settings', {})
        sentences_per_line = styletts2_settings.get('sentences_per_voice_line', 3)
    else:
        # Default fallback
        sentences_per_line = 1
        
    log(f"Using {sentences_per_line} sentences per voice line for {provider_name} provider", "🔊")
    
    # Process sentences in groups
    completed = 0
    skipped = 0
    total_groups = (total_sentences + sentences_per_line - 1) // sentences_per_line
    log(f"Processing {total_groups} voice lines...")
    
    # Track generation time for statistics
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(total=total_sentences, desc="Generating voice lines", unit="sentence")
    
    # Variables for time estimation
    processed_sentences = 0
    generation_times = []
    
    for group_idx in range(total_groups):
        # Get sentences for this group
        start_idx = group_idx * sentences_per_line
        end_idx = min(start_idx + sentences_per_line, total_sentences)
        group_sentences = story['sentences'][start_idx:end_idx]
        
        # Create output filename with zero-padded index
        output_filename = f"{start_idx:04d}.wav"
        output_path = os.path.join(output_dir_path, output_filename)
        
        # Check if the file already exists
        if os.path.exists(output_path) and not force_regenerate:
            log(f"Skipping group {group_idx+1}/{total_groups}: File already exists", "⏩")
            skipped += len(group_sentences)
            # Update progress bar for skipped sentences
            pbar.update(len(group_sentences))
            continue
        
        # Show progress percentage
        progress_pct = (start_idx / total_sentences) * 100
        
        # Clean each sentence individually first
        cleaned_sentences = [preprocess_text_for_tts(s) for s in group_sentences]
        
        # Combine sentences and ensure proper spacing
        combined_text = ' '.join(s.strip() for s in cleaned_sentences if s.strip())
        combined_text = preprocess_text_for_tts(combined_text)  # One final cleanup
        
        log(f"Processing group {group_idx+1}/{total_groups} ({progress_pct:.1f}%):", "🔊")
        log(f"Text: \"{combined_text}\"", "📝")
        
        # Track generation time for this group
        group_start_time = time.time()
        
        # Generate voice line for combined sentences
        if generate_voice_line(combined_text, output_path, voice_options):
            # Calculate generation time for this group
            group_time = time.time() - group_start_time
            generation_times.append(group_time / len(group_sentences))
            
            # Update progress bar
            pbar.update(len(group_sentences))
            processed_sentences += len(group_sentences)
            
            # Calculate and display estimated time remaining
            if len(generation_times) > 0:
                avg_time = sum(generation_times) / len(generation_times)
                remaining_sentences = total_sentences - processed_sentences - skipped
                est_remaining_time = remaining_sentences * avg_time
                
                # Format estimated time remaining
                est_time_str = str(datetime.timedelta(seconds=int(est_remaining_time)))
                
                # Update progress bar description
                pbar.set_description(f"Generating voice lines (ETA: {est_time_str})")
            
            log(f"Generated: {output_filename}", "✅")
            completed += len(group_sentences)
            
            # Apply audio effects if any are enabled
            effects_applied = False
            
            # Apply audio effects if enabled
            if tempo_setting or echo_setting:
                # Prepare settings dictionaries
                tempo_settings = {
                    'enabled': tempo_setting,
                    'factor': tempo_factor
                }
                
                echo_settings = {
                    'enabled': echo_setting,
                    'delay': echo_delay,
                    'decay': echo_decay
                }
                
                log(f"Applying audio effects...", "🎧")
                success, original_duration, new_duration = apply_audio_effects(
                    output_path, output_path, tempo_settings, echo_settings
                )
                
                if success:
                    log(f"Applied effects: {output_filename} ({original_duration:.2f}s → {new_duration:.2f}s)", "✅")
                    effects_applied = True
                else:
                    log(f"Failed to apply audio effects: {output_filename}", "⚠️")
            
            # Normalize audio if enabled (after effects)
            if normalize_audio_setting and os.path.exists(output_path):
                log(f"Normalizing audio to {target_db} dB...", "🔊")
                success, _, _ = normalize_audio(output_path, None, target_db)
                if success:
                    log(f"Normalized: {output_filename}", "✅")
                else:
                    log(f"Failed to normalize: {output_filename}", "⚠️")
        else:
            log(f"Failed to generate: {output_filename}", "❌")
    
    # Close the progress bar
    pbar.close()
    
    # Calculate total time taken
    total_time = time.time() - start_time
    time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    # Show summary for this story
    log(f"\nSummary for '{story['topic_title']}':" , "📊")
    log(f"  - Total sentences: {total_sentences}", "📊")
    log(f"  - Generated: {completed}", "📊")
    log(f"  - Skipped: {skipped}", "📊")
    log(f"  - Failed: {total_sentences - completed - skipped}", "📊")
    log(f"  - Total time: {time_str}", "📊")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate voice lines for stories')
    parser.add_argument('-s', '--story', type=str,
                        help='Process only the specified story file (filename only, not full path)')
    parser.add_argument('-v', '--voice', type=int,
                        help='Force the use of a specific voice sample by index (0, 1, 2, etc.)')
    parser.add_argument('-p', '--provider', choices=['zonos', 'orpheus', 'styletts2'],
                        help='TTS provider to use (overrides config)')
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
    parser.add_argument('--tempo', action='store_true',
                        help='Force tempo manipulation even if disabled in config')
    parser.add_argument('--no-tempo', action='store_true',
                        help='Disable tempo manipulation even if enabled in config')
    parser.add_argument('--tempo-factor', type=float,
                        help='Tempo change factor (< 1.0 slows down, > 1.0 speeds up)')
    parser.add_argument('--echo', action='store_true',
                        help='Force echo effect even if disabled in config')
    parser.add_argument('--no-echo', action='store_true',
                        help='Disable echo effect even if enabled in config')
    parser.add_argument('--echo-delay', type=float,
                        help='Echo delay in seconds')
    parser.add_argument('--echo-decay', type=float,
                        help='Echo decay factor (0-1)')
    return parser.parse_args()

def main():
    """Main function to process all stories."""
    args = parse_arguments()
    
    # Load global config variables
    global CONFIG, TTS_PROVIDER, STORIES_DIR, VOICE_LINES_DIR
    CONFIG = load_config()
    
    # Override provider if specified in command line arguments
    voice_config = CONFIG.get('voice', {})
    if args.provider:
        log(f"Overriding TTS provider from config with: {args.provider}", "🔄")
        voice_config['provider'] = args.provider
    
    # Initialize the appropriate TTS provider
    TTS_PROVIDER = get_tts_provider(voice_config)
    
    # Set directory names
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
        
        # Determine tempo settings
        tempo_setting = None
        if args.tempo:
            tempo_setting = True
        elif args.no_tempo:
            tempo_setting = False
        
        # Determine echo settings
        echo_setting = None
        if args.echo:
            echo_setting = True
        elif args.no_echo:
            echo_setting = False
            
        # Process the story
        process_story(
            story_file, 
            args.force, 
            normalize_audio_setting, 
            args.target_db, 
            args.voice,
            tempo_setting,
            args.tempo_factor,
            echo_setting,
            args.echo_delay,
            args.echo_decay
        )
    
    log("\nVoice line generation complete!", "🎉")
    return 0

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Process interrupted by user (Ctrl+C)", "⚠️")
        log("Exiting gracefully...", "🛑")
        sys.exit(130)  # Standard exit code for SIGINT
