#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SleepTeller - Video Composition Module

This script combines the generated audio with a looping background video or image
to create a final video file ready for playback.
"""

import os
import sys
import yaml
import argparse
import glob
from pathlib import Path
import time
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, CompositeAudioClip

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
STORIES_DIR = None
AUDIO_DIR = None
VIDEOS_DIR = None
PROJECT_NAME = None

def update_directories():
    """Update directory paths based on the loaded configuration."""
    global STORIES_DIR, AUDIO_DIR, VIDEOS_DIR, PROJECT_NAME
    
    PROJECT_NAME = CONFIG.get('project_name', 'default')
    
    # Create output directory structure
    output_dir = os.path.join(PROJECT_DIR, "output", PROJECT_NAME)
    STORIES_DIR = os.path.join(output_dir, "stories")
    AUDIO_DIR = os.path.join(output_dir, "audio")
    VIDEOS_DIR = os.path.join(output_dir, "videos")
    
    # Create directories if they don't exist
    os.makedirs(STORIES_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    
    log(f"Project: {PROJECT_NAME}", "📂")
    log(f"Stories directory: {STORIES_DIR}", "📝")
    log(f"Audio directory: {AUDIO_DIR}", "🔊")
    log(f"Videos directory: {VIDEOS_DIR}", "🎬")

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

# ─────────────────────────────────────────────────────
# 🎬 VIDEO PROCESSING
# ─────────────────────────────────────────────────────
def resize_video(clip, target_resolution):
    """Resize a video clip to the target resolution while maintaining aspect ratio.
    
    Args:
        clip: The video clip to resize
        target_resolution: Tuple of (width, height)
    
    Returns:
        Resized video clip
    """
    target_width, target_height = target_resolution
    
    # Get the current dimensions
    current_width, current_height = clip.size
    
    # Calculate the scaling factors
    width_ratio = target_width / current_width
    height_ratio = target_height / current_height
    
    # Use the smaller ratio to ensure the video fits within the target resolution
    scale_factor = min(width_ratio, height_ratio)
    
    # Calculate the new dimensions
    new_width = int(current_width * scale_factor)
    new_height = int(current_height * scale_factor)
    
    # Resize the clip
    resized_clip = clip.resize((new_width, new_height))
    
    # Create a black background clip
    background = CompositeVideoClip([resized_clip.set_position("center")], 
                                    size=target_resolution)
    
    return background

def find_latest_audio():
    """Find the most recently created audio file.
    
    Returns:
        Path to the latest audio file, or None if no audio found
    """
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
    
    if not audio_files:
        return None
    
    # Sort by modification time (newest first)
    latest_audio = max(audio_files, key=os.path.getmtime)
    return latest_audio

def create_video(audio_path, background_path=None, output_path=None, add_music=False, music_file=None, music_volume=0.2):
    """Create a video by combining audio with a background video or image.
    
    Args:
        audio_path: Path to the audio file
        background_path: Path to the background video or image file
        output_path: Path to save the output video
        add_music: Whether to add background music
        music_file: Path to the background music file
        music_volume: Volume of the background music (0.0 to 1.0)
    
    Returns:
        Path to the created video file
    """
    # Load the audio file
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    
    log(f"Audio duration: {audio_duration:.2f} seconds", "⏱️")
    
    # If no background path is provided, use the one from config
    if background_path is None:
        background_path = CONFIG.get('video', {}).get('background_loop')
    
    if not background_path or not os.path.exists(background_path):
        log(f"Background file not found: {background_path}", "❌")
        log("Please specify a valid background file in the config or as a parameter", "💡")
        return None
    
    # Determine if the background is a video or an image
    is_image = background_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
    
    # Get the target resolution from config or use default
    resolution_str = CONFIG.get('video', {}).get('resolution', '1920x1080')
    width, height = map(int, resolution_str.split('x'))
    target_resolution = (width, height)
    
    # Create the background clip
    if is_image:
        log(f"Using image as background: {background_path}", "🖼️")
        background_clip = ImageClip(background_path).set_duration(audio_duration)
        background_clip = resize_video(background_clip, target_resolution)
    else:
        log(f"Using video as background: {background_path}", "🎞️")
        background_clip = VideoFileClip(background_path)
        
        # Loop the background video if it's shorter than the audio
        if background_clip.duration < audio_duration:
            log(f"Background video ({background_clip.duration:.2f}s) is shorter than audio ({audio_duration:.2f}s), will loop", "🔄")
            background_clip = background_clip.loop(duration=audio_duration)
        else:
            # Trim the background video if it's longer than the audio
            background_clip = background_clip.subclip(0, audio_duration)
        
        background_clip = resize_video(background_clip, target_resolution)
    
    # Create the final audio track
    if add_music and music_file and os.path.exists(music_file):
        log(f"Adding background music: {music_file}", "🎵")
        music_clip = AudioFileClip(music_file)
        
        # Loop the music if it's shorter than the audio
        if music_clip.duration < audio_duration:
            music_clip = music_clip.loop(duration=audio_duration)
        else:
            # Trim the music if it's longer than the audio
            music_clip = music_clip.subclip(0, audio_duration)
        
        # Adjust the volume of the music
        music_clip = music_clip.volumex(music_volume)
        
        # Combine the narration and music
        final_audio = CompositeAudioClip([audio_clip, music_clip])
    else:
        final_audio = audio_clip
    
    # Set the audio of the background clip
    video_with_audio = background_clip.set_audio(final_audio)
    
    # If no output path is provided, create one based on the audio filename
    if output_path is None:
        audio_filename = os.path.basename(audio_path)
        audio_name = os.path.splitext(audio_filename)[0]
        output_path = os.path.join(VIDEOS_DIR, f"{audio_name}.mp4")
    
    # Get the output format from config or use default
    output_format = CONFIG.get('video', {}).get('output_format', 'mp4')
    
    # Write the final video
    log(f"Rendering final video...", "🎬")
    video_with_audio.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile=os.path.join(VIDEOS_DIR, "temp_audio.m4a"),
        remove_temp=True,
        fps=30
    )
    
    log(f"Video created successfully: {output_path}", "✅")
    return output_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SleepTeller - Video Composition Module")
    
    parser.add_argument("-c", "--config", default="configs/sample.yaml", help="Path to the config file (default: configs/sample.yaml)")
    parser.add_argument("-a", "--audio", help="Path to the audio file (if not specified, uses the latest)")
    parser.add_argument("-b", "--background", help="Path to the background video or image (overrides config)")
    parser.add_argument("-m", "--music", help="Path to the background music file (overrides config)")
    parser.add_argument("--no-music", action="store_true", help="Disable background music even if specified in config")
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    global CONFIG
    
    args = parse_args()
    
    # Load configuration
    CONFIG = load_config(args.config)
    
    # Update directories based on config
    update_directories()
    
    # Get the audio file
    audio_path = args.audio
    if not audio_path:
        audio_path = find_latest_audio()
        if not audio_path:
            log("No audio files found. Please generate audio first using make_audio.py", "❌")
            return None
    
    log(f"Using audio file: {os.path.basename(audio_path)}", "🔊")
    
    # Get the background file
    background_path = args.background or CONFIG.get('video', {}).get('background_loop')
    
    if not background_path:
        log("No background file specified in config or command line", "❌")
        return None
    
    log(f"Using background: {background_path}", "🎬")
    
    # Determine if we should add background music
    add_music = not args.no_music and CONFIG.get('video', {}).get('add_music', False)
    
    # Get the music file
    music_file = args.music or CONFIG.get('video', {}).get('music_file')
    music_volume = CONFIG.get('video', {}).get('music_volume', 0.2)
    
    if add_music and not music_file:
        log("Background music is enabled but no music file specified", "⚠️")
        add_music = False
    
    # Create the video
    video_path = create_video(
        audio_path=audio_path,
        background_path=background_path,
        add_music=add_music,
        music_file=music_file,
        music_volume=music_volume
    )
    
    if video_path:
        log(f"Video creation complete!", "🎉")
        log(f"Video saved to: {video_path}", "🎬")
        return video_path
    else:
        log("Video creation failed", "❌")
        return None

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nProcess interrupted by user", "⚠️")
        log("Exiting gracefully...", "🛑")
        sys.exit(130)  # Standard exit code for SIGINT
