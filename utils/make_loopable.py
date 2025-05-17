#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SleepTeller - Video Loop Utility

This script processes videos to make them more loopable by crossfading
the end of the video with a segment from the beginning.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip

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
        return {}
    except yaml.YAMLError as e:
        log(f"Error parsing config.yaml: {e}", "❌")
        return {}

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist.
    
    Args:
        directory: Path to the directory to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        log(f"Created directory: {directory}", "📁")

def get_video_files(input_dir):
    """Get all video files from the input directory.
    
    Args:
        input_dir: Path to the input directory
        
    Returns:
        List of paths to video files
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(Path(input_dir).glob(f'*{ext}')))
    
    return video_files

def make_video_loopable(input_path, output_path, crossfade_duration=5, encoding_settings=None):
    """Make a video loopable by crossfading the end with a segment from the beginning.
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the output video
        crossfade_duration: Duration of the crossfade segment in seconds
        encoding_settings: Dictionary of encoding settings from config
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the video
        video = VideoFileClip(str(input_path))
        
        # Get video duration
        duration = video.duration
        
        if duration <= crossfade_duration * 2:
            log(f"Video {input_path.name} is too short for a {crossfade_duration}s crossfade", "⚠️")
            return False
        
        # Extract the beginning segment that will be used for crossfading
        begin_segment = video.subclip(0, crossfade_duration)
        
        # Extract the end segment that will be crossfaded with the beginning
        end_segment = video.subclip(duration - crossfade_duration, duration)
        
        # Create the main part of the video (excluding the crossfade segments)
        main_video = video.subclip(crossfade_duration, duration - crossfade_duration)
        
        # Create a crossfade between the end and beginning segments
        # We'll keep the end segment at full opacity and only fade in the beginning segment on top
        crossfade_clips = []
        
        # Add the end segment at full opacity (no fadeout)
        crossfade_clips.append(end_segment)
        
        # Add the beginning segment with a fadein effect, positioned at the same time as end_segment
        # This will gradually increase opacity from 0% to 100%, blending on top of the end segment
        begin_with_fade = begin_segment.set_start(0).crossfadein(crossfade_duration)
        crossfade_clips.append(begin_with_fade)
        
        # Combine the crossfaded segments
        crossfade_part = CompositeVideoClip(crossfade_clips, size=video.size)
        
        # Create the final loopable video by concatenating:
        # 1. The crossfade between end and beginning
        # 2. The main part of the video
        final_clips = [crossfade_part, main_video.set_start(crossfade_duration)]
        final_video = CompositeVideoClip(final_clips, size=video.size)
        
        # Set default encoding settings if none provided
        if not encoding_settings:
            encoding_settings = {
                'video_codec': 'libx264',
                'video_bitrate': '4M',
                'audio_codec': 'aac',
                'audio_bitrate': '128k',
                'preset': 'medium',
                'threads': 0,
                'fps': video.fps,
                'crf': 23
            }
        
        # Use fps from video if not specified in settings
        if 'fps' not in encoding_settings:
            encoding_settings['fps'] = video.fps
            
        # Log encoding settings
        log(f"Encoding with: {encoding_settings['video_codec']} @ {encoding_settings['video_bitrate']}, CRF: {encoding_settings.get('crf', 'auto')}", "🎬")
        
        # Write the output video with the specified encoding settings
        final_video.write_videofile(
            str(output_path),
            codec=encoding_settings['video_codec'],
            bitrate=encoding_settings['video_bitrate'],
            audio_codec=encoding_settings['audio_codec'],
            audio_bitrate=encoding_settings['audio_bitrate'],
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            preset=encoding_settings.get('preset', 'medium'),
            threads=encoding_settings.get('threads', 0),
            fps=encoding_settings['fps']
        )
        
        # Close the clips to release resources
        video.close()
        begin_segment.close()
        end_segment.close()
        main_video.close()
        crossfade_part.close()
        final_video.close()
        
        log(f"Created loopable video: {output_path.name}", "✅")
        return True
        
    except Exception as e:
        log(f"Error processing {input_path.name}: {str(e)}", "❌")
        return False

def process_videos(input_dir, output_dir, crossfade_duration):
    """Process all videos in the input directory.
    
    Args:
        input_dir: Path to the input directory
        output_dir: Path to the output directory
        crossfade_duration: Duration of the crossfade segment in seconds
        
    Returns:
        Number of successfully processed videos
    """
    # Ensure output directory exists
    ensure_dir_exists(output_dir)
    
    # Load config for encoding settings
    config = load_config()
    encoding_settings = None
    
    # Get encoding settings from config if available
    if config and 'video' in config and 'encoding' in config['video']:
        encoding_settings = config['video']['encoding']
        log(f"Using encoding settings from config.yaml", "⚙️")
    else:
        log("No encoding settings found in config.yaml, using defaults", "⚙️")
    
    # Get all video files
    video_files = get_video_files(input_dir)
    
    if not video_files:
        log(f"No video files found in {input_dir}", "⚠️")
        return 0
    
    log(f"Found {len(video_files)} video files to process", "🎬")
    
    # Process each video
    success_count = 0
    for video_file in video_files:
        output_path = Path(output_dir) / f"loopable_{video_file.name}"
        
        log(f"Processing {video_file.name}...", "🔄")
        if make_video_loopable(video_file, output_path, crossfade_duration, encoding_settings):
            success_count += 1
    
    return success_count

def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Make videos loopable by crossfading the end with the beginning")
    parser.add_argument("input_dir", help="Directory containing input videos")
    parser.add_argument("output_dir", help="Directory to save processed videos")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of the crossfade segment in seconds (default: 5.0)")
    parser.add_argument("--bitrate", help="Override video bitrate (e.g., '4M')")
    parser.add_argument("--preset", choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'], 
                        help="Encoding preset (speed/quality tradeoff)")
    parser.add_argument("--crf", type=int, help="Constant Rate Factor (0-51, lower = better quality)")
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        log(f"Error: Input directory '{args.input_dir}' does not exist", "❌")
        return 1
    
    log(f"Making videos loopable with {args.duration}s crossfade", "🔄")
    log(f"Input directory: {args.input_dir}", "📂")
    log(f"Output directory: {args.output_dir}", "📂")
    
    # Process videos
    success_count = process_videos(args.input_dir, args.output_dir, args.duration)
    
    # Report results
    if success_count > 0:
        log(f"Successfully processed {success_count} videos", "✅")
    else:
        log("No videos were successfully processed", "⚠️")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Operation cancelled by user", "🛑")
        sys.exit(1)
