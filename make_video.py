#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SleepTeller - Video Composition Module

This script combines the generated voice lines with a looping background video
and background music to create a final sleep story video ready for playback.
"""

import os
import sys
import yaml
import argparse
import glob
import random
import random
from pathlib import Path
import time
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips, CompositeVideoClip, CompositeAudioClip

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
STORIES_DIR = None
VOICE_LINES_DIR = None
VIDEOS_OUTPUT_DIR = None
LIB_VIDEOS_DIR = None
LIB_MUSIC_DIR = None

def update_directories():
    """Update directory paths for input and output files."""
    global STORIES_DIR, VOICE_LINES_DIR, VIDEOS_OUTPUT_DIR, LIB_VIDEOS_DIR, LIB_MUSIC_DIR
    
    # Input directories
    STORIES_DIR = os.path.join(PROJECT_DIR, "output", "stories")
    VOICE_LINES_DIR = os.path.join(PROJECT_DIR, "output", "voice_lines")
    
    # Output directory
    VIDEOS_OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "videos")
    
    # Library directories
    LIB_VIDEOS_DIR = os.path.join(PROJECT_DIR, "lib", "videos")
    LIB_MUSIC_DIR = os.path.join(PROJECT_DIR, "lib", "music")
    
    # Create directories if they don't exist
    os.makedirs(STORIES_DIR, exist_ok=True)
    os.makedirs(VIDEOS_OUTPUT_DIR, exist_ok=True)
    
    log(f"Stories directory: {STORIES_DIR}", "📝")
    log(f"Voice lines directory: {VOICE_LINES_DIR}", "🔊")
    log(f"Videos output directory: {VIDEOS_OUTPUT_DIR}", "🎥")
    log(f"Background videos directory: {LIB_VIDEOS_DIR}", "🎥")
    log(f"Background music directory: {LIB_MUSIC_DIR}", "🌻")

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

def get_story_files():
    """Get all story YAML files.
    
    Returns:
        List of paths to story files
    """
    story_files = glob.glob(os.path.join(STORIES_DIR, "*.yaml"))
    return story_files

def get_voice_lines(story_name):
    """Get all voice line files for a specific story.
    
    Args:
        story_name: Name of the story (directory name in voice_lines)
        
    Returns:
        List of paths to voice line files, sorted by number
    """
    voice_lines_path = os.path.join(VOICE_LINES_DIR, story_name)
    if not os.path.exists(voice_lines_path):
        return []
    
    # Get all WAV files
    voice_files = glob.glob(os.path.join(voice_lines_path, "*.wav"))
    
    # Sort by filename (which should be numeric)
    voice_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    return voice_files

def get_random_background_video():
    """Get a random background video from the library.
    
    Returns:
        Path to a random video file, or None if no videos found
    """
    video_files = glob.glob(os.path.join(LIB_VIDEOS_DIR, "*.mp4"))
    video_files.extend(glob.glob(os.path.join(LIB_VIDEOS_DIR, "*.mov")))
    video_files.extend(glob.glob(os.path.join(LIB_VIDEOS_DIR, "*.avi")))
    
    if not video_files:
        return None
    
    return random.choice(video_files)

def get_random_background_music():
    """Get a random background music file from the library.
    
    Returns:
        Path to a random music file, or None if no music found
    """
    music_files = glob.glob(os.path.join(LIB_MUSIC_DIR, "*.mp3"))
    music_files.extend(glob.glob(os.path.join(LIB_MUSIC_DIR, "*.wav")))
    
    if not music_files:
        return None
    
    return random.choice(music_files)

def create_video_from_story(story_name, output_path=None, quality=1080, force=False):
    """Create a video for a specific story by combining voice lines with background video and music.
    
    Args:
        story_name: Name of the story (without extension)
        output_path: Path to save the output video (default: output/videos/{story_name}.mp4)
        quality: Target video height (e.g., 720 for 720p, 1080 for 1080p)
        force: Whether to force regeneration even if the video already exists
        
    Returns:
        Path to the created video file, or None if creation failed
    """
    # Determine the output path if not provided
    if output_path is None:
        output_path = os.path.join(VIDEOS_OUTPUT_DIR, f"{story_name}.mp4")
    
    # Check if the video already exists and skip if not forcing regeneration
    if os.path.exists(output_path) and not force:
        log(f"Video already exists: {output_path}", "⏩")
        log("Use --force to regenerate", "💡")
        return output_path
    
    # Get all voice lines for this story
    voice_files = get_voice_lines(story_name)
    if not voice_files:
        log(f"No voice lines found for story: {story_name}", "❌")
        return None
    
    log(f"Found {len(voice_files)} voice lines for story: {story_name}", "🔊")
    
    # Get delay settings from config
    start_delay = CONFIG.get('video', {}).get('start_delay', 2.0)
    end_delay = CONFIG.get('video', {}).get('end_delay', 10.0)
    line_delay_min = CONFIG.get('video', {}).get('line_delay_min', 0.65)
    line_delay_max = CONFIG.get('video', {}).get('line_delay_max', 0.9)
    
    log(f"Using delays: start={start_delay}s, between lines={line_delay_min}-{line_delay_max}s, end={end_delay}s", "⏱️")
    
    # Load all voice line audio clips and trim the end slightly to prevent clicks
    voice_clips = [AudioFileClip(file).subclip(0, -0.05) for file in voice_files]
    
    # Create silent clips for delays
    from moviepy.audio.AudioClip import AudioClip
    start_silence = AudioClip(make_frame=lambda t: 0, duration=start_delay)
    end_silence = AudioClip(make_frame=lambda t: 0, duration=end_delay)
    
    # Insert silence between voice clips with random delays
    clips_with_delays = [start_silence]  # Start with initial delay
    
    for i, clip in enumerate(voice_clips):
        clips_with_delays.append(clip)
        # Add random delay between clips (except after the last one)
        if i < len(voice_clips) - 1:
            random_delay = random.uniform(line_delay_min, line_delay_max)
            line_silence = AudioClip(make_frame=lambda t: 0, duration=random_delay)
            clips_with_delays.append(line_silence)
    
    # Add end delay
    clips_with_delays.append(end_silence)
    
    # Concatenate all clips with delays
    log("Concatenating voice lines with delays...", "🔊")
    full_audio = concatenate_audioclips(clips_with_delays)
    audio_duration = full_audio.duration
    
    log(f"Total audio duration: {audio_duration:.2f} seconds", "⏱️")
    
    # Get a random background video
    background_path = get_random_background_video()
    if not background_path:
        log("No background videos found in lib/videos", "❌")
        return None
    
    log(f"Using background video: {os.path.basename(background_path)}", "🎥")
    
    # Get a random background music
    music_file = get_random_background_music()
    if not music_file:
        log("No background music found in lib/music", "⚠️")
    else:
        log(f"Using background music: {os.path.basename(music_file)}", "🌻")
    
    # Calculate target resolution based on desired height while maintaining 16:9 aspect ratio
    target_height = quality  # e.g., 720, 1080
    target_width = int(target_height * 16 / 9)  # maintain 16:9 aspect ratio
    target_resolution = (target_width, target_height)
    
    log(f"Target resolution: {target_width}x{target_height}", "📺")
    
    # Load the background video without audio
    background_clip = VideoFileClip(background_path, audio=False)
    log("Loading background video without audio", "🔇")
    
    # Get crossfade duration from config (default 0.0 = disabled)
    crossfade_duration = CONFIG.get('video', {}).get('crossfade_duration', 0.0)
    
    # Loop the background video if it's shorter than the audio
    if background_clip.duration < audio_duration:
        log(f"Background video ({background_clip.duration:.2f}s) is shorter than audio ({audio_duration:.2f}s), will loop", "🔄")
        
        # If crossfade is enabled (duration > 0), create a crossfade loop
        if crossfade_duration > 0 and background_clip.duration > crossfade_duration * 2:
            log(f"Applying crossfade of {crossfade_duration:.2f}s for smoother looping", "🔀")
            
            # Create a clip for the beginning portion to be used at the end for crossfade
            start_clip = background_clip.subclip(0, crossfade_duration)
            
            # Create a clip for the end portion to be used at the beginning for crossfade
            end_clip = background_clip.subclip(background_clip.duration - crossfade_duration, background_clip.duration)
            
            # Create the main clip without the end portion that will be crossfaded
            main_clip = background_clip.subclip(0, background_clip.duration - crossfade_duration)
            
            # Calculate how many full loops we need (not including crossfades)
            remaining_duration = audio_duration - main_clip.duration
            num_full_loops = int(remaining_duration / main_clip.duration)
            
            # Create a list to hold all the clips
            clips = []
            
            # Add the first clip with crossfade from the end
            clips.append(CompositeVideoClip([
                end_clip.set_start(0),
                main_clip.set_start(0)
            ]).crossfadeout(crossfade_duration))
            
            # Add full loops as needed
            current_time = main_clip.duration
            for i in range(num_full_loops):
                clips.append(main_clip.set_start(current_time))
                current_time += main_clip.duration
            
            # Add the final clip with crossfade to the beginning if needed
            remaining_time = audio_duration - current_time
            if remaining_time > 0:
                final_clip = main_clip.subclip(0, min(remaining_time + crossfade_duration, main_clip.duration))
                clips.append(final_clip.set_start(current_time).crossfadein(crossfade_duration))
            
            # Combine all clips
            background_clip = CompositeVideoClip(clips, size=background_clip.size).set_duration(audio_duration)
        else:
            # Use the standard loop method if crossfade is disabled
            background_clip = background_clip.loop(duration=audio_duration)
    else:
        # Trim the background video if it's longer than the audio
        log(f"Background video ({background_clip.duration:.2f}s) is longer than audio ({audio_duration:.2f}s), will trim", "✂️")
        background_clip = background_clip.subclip(0, audio_duration)
    
    # Resize the background video to the target resolution
    background_clip = resize_video(background_clip, target_resolution)
    
    # Create the final audio track with background music if available
    if music_file and os.path.exists(music_file):
        log(f"Adding background music: {os.path.basename(music_file)}", "🎵")
        music_clip = AudioFileClip(music_file)
        
        # Loop the music if it's shorter than the audio
        if music_clip.duration < audio_duration:
            log(f"Music ({music_clip.duration:.2f}s) is shorter than audio ({audio_duration:.2f}s), will loop", "🔄")
            
            # Create a custom looped music clip by concatenating multiple copies
            music_duration = music_clip.duration
            num_loops_needed = int(audio_duration / music_duration) + 1
            
            # Create a list of music clips to concatenate
            music_clips = []
            for i in range(num_loops_needed):
                # For all except the last clip, use the full duration
                if i < num_loops_needed - 1:
                    music_clips.append(music_clip)
                else:
                    # For the last clip, only use what's needed to reach the target duration
                    remaining_duration = audio_duration - (i * music_duration)
                    if remaining_duration > 0:
                        music_clips.append(music_clip.subclip(0, min(remaining_duration, music_duration)))
            
            # Concatenate all music clips
            music_clip = concatenate_audioclips(music_clips)
        else:
            # Trim the music if it's longer than the audio
            log(f"Music ({music_clip.duration:.2f}s) is longer than audio ({audio_duration:.2f}s), will trim", "✂️")
            music_clip = music_clip.subclip(0, audio_duration)
        
        # Get music volume from config (default 33%)
        music_volume = CONFIG.get('video', {}).get('music_volume', 0.33)
        log(f"Setting music volume to {music_volume * 100:.0f}% of voice volume", "🔊")
        music_clip = music_clip.volumex(music_volume)
        
        # Combine the voice lines and music
        final_audio = CompositeAudioClip([full_audio, music_clip])
    else:
        final_audio = full_audio
    
    # Set the audio of the background clip
    video_with_audio = background_clip.set_audio(final_audio)
    
    # Apply fade in and fade out effects if configured
    fade_in_duration = CONFIG.get('video', {}).get('fade_in_duration', 0.0)
    fade_out_duration = CONFIG.get('video', {}).get('fade_out_duration', 0.0)
    
    if fade_in_duration > 0:
        log(f"Applying fade in from black: {fade_in_duration:.2f}s", "🌅")
        video_with_audio = video_with_audio.fadein(fade_in_duration)
        
    if fade_out_duration > 0:
        log(f"Applying fade out to black: {fade_out_duration:.2f}s", "🌆")
        video_with_audio = video_with_audio.fadeout(fade_out_duration)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the final video
    log(f"Rendering final video for {story_name}...", "🎥")
    
    # Get encoding settings from config
    encoding_config = CONFIG.get('video', {}).get('encoding', {})
    video_codec = encoding_config.get('video_codec', 'libx264')
    video_bitrate = encoding_config.get('video_bitrate', '2M')
    audio_codec = encoding_config.get('audio_codec', 'aac')
    audio_bitrate = encoding_config.get('audio_bitrate', '128k')
    fps = encoding_config.get('fps', 24)
    threads = encoding_config.get('threads', 0)
    preset = encoding_config.get('preset', 'ultrafast')
    tune = encoding_config.get('tune', 'fastdecode')
    crf = encoding_config.get('crf', 28)
    
    log(f"Using codec: {video_codec}, bitrate: {video_bitrate}, preset: {preset}, tune: {tune}", "🎬")
    
    # Prepare ffmpeg parameters for preset, tune and CRF
    ffmpeg_params = [
        f"-preset", preset,
        f"-tune", tune,
        f"-crf", str(crf)
    ]
    
    video_with_audio.write_videofile(
        output_path,
        codec=video_codec,
        bitrate=video_bitrate,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        temp_audiofile=os.path.join(VIDEOS_OUTPUT_DIR, "temp_audio.m4a"),
        remove_temp=True,
        fps=fps,
        threads=threads,
        ffmpeg_params=ffmpeg_params
    )
    
    log(f"Video created successfully: {output_path}", "✅")
    return output_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SleepTeller - Video Composition Module")
    
    parser.add_argument("-s", "--story", help="Process only the specified story (filename without extension)")
    parser.add_argument("-f", "--force", action="store_true", help="Force regeneration even for existing videos")
    parser.add_argument("-q", "--quality", type=int, default=720, help="Video quality as frame height (e.g., 720 for 720p, 1080 for 1080p)")
    
    return parser.parse_args()

def cleanup_moviepy():
    """Clean up MoviePy resources to prevent FFMPEG reader errors."""
    try:
        import gc
        gc.collect()
    except Exception as e:
        log(f"Cleanup warning (non-critical): {str(e)}", "⚠️")

def main():
    """Main entry point for the script."""
    global CONFIG
    
    args = parse_args()
    
    # Validate quality (common resolutions: 480p, 720p, 1080p, 1440p, 2160p)
    valid_resolutions = [360, 480, 720, 1080, 1440, 2160]
    if args.quality <= 0:
        log("Quality (frame height) must be a positive number", "❌")
        return 1
    if args.quality > 2160:
        log(f"Quality {args.quality}p is unusually high (>2160p)", "⚠️")

    # Load configuration
    CONFIG = load_config()
    
    # Update directories
    update_directories()
    
    # Check if lib directories exist and have content
    if not os.path.exists(LIB_VIDEOS_DIR) or not os.listdir(LIB_VIDEOS_DIR):
        log(f"No background videos found in {LIB_VIDEOS_DIR}", "❌")
        log("Please add some video files to the lib/videos directory", "💡")
        return 1
    
    if not os.path.exists(LIB_MUSIC_DIR) or not os.listdir(LIB_MUSIC_DIR):
        log(f"No background music found in {LIB_MUSIC_DIR}", "⚠️")
        log("You may want to add some music files to the lib/music directory", "💡")
    
    # Get all story files or filter to a specific one
    all_story_files = get_story_files()
    if not all_story_files:
        log("No story files found in output/stories", "❌")
        log("Please generate stories first using make_story.py", "💡")
        return 1
    
    # Filter to a specific story if requested
    if args.story:
        story_files = [f for f in all_story_files if os.path.splitext(os.path.basename(f))[0] == args.story]
        if not story_files:
            log(f"Story not found: {args.story}", "❌")
            return 1
    else:
        story_files = all_story_files
    
    log(f"Processing {len(story_files)} stories", "📝")
    
    # Process each story
    successful = 0
    for story_file in story_files:
        story_name = os.path.splitext(os.path.basename(story_file))[0]
        log(f"\nProcessing story: {story_name}", "📝")
        
        # Create video for this story
        video_path = create_video_from_story(
            story_name=story_name,
            quality=args.quality,
            force=args.force
        )
        
        if video_path:
            log(f"Video created successfully: {video_path}", "✅")
            successful += 1
    
    # Print summary
    log(f"\nVideo creation complete! {successful}/{len(story_files)} videos created successfully", "🎉")
    
    # Clean up MoviePy resources to prevent FFMPEG reader errors
    cleanup_moviepy()
    
    return 0

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nProcess interrupted by user", "⚠️")
        log("Exiting gracefully...", "🔴")
    finally:
        # Ensure cleanup happens even if there's an exception
        cleanup_moviepy()
        sys.exit(130)  # Standard exit code for SIGINT
