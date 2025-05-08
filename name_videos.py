#!/usr/bin/env python3
"""
Usage:
    python name_videos.py [-m MODEL] [-f FRAMES] [--min MIN_LENGTH] [--max MAX_LENGTH] [-r]

This script processes videos in the lib/videos folder by:
1. Extracting frames from each video
2. Getting descriptions for each frame using an LLM
3. Generating a summary name for the video based on the frame descriptions
4. Renaming the video file with the generated name

Arguments:
    -m, --model MODEL       Model name to use for image analysis (default: gemma-3-4b-it)
    -f, --frames FRAMES     Number of frames to extract per video (default: 8)
    --min MIN_LENGTH        Minimum filename length (default: 32)
    --max MAX_LENGTH        Maximum filename length (default: 128)
"""

import argparse
import os
import sys
import shutil
import tempfile
from pathlib import Path
import lmstudio as lms
from pydantic import BaseModel
import mimetypes
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image
import yaml

# Get the absolute path of the project directory
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
VIDEOS_DIR = os.path.join(PROJECT_DIR, "lib", "videos")
TEMP_DIR = os.path.join(PROJECT_DIR, "temp")


class FrameDescription(BaseModel):
    """Model for frame description response from LLM."""
    description: str


class VideoSummary(BaseModel):
    """Model for video summary response from LLM."""
    suggested_filename: str


def is_video_file(file_path):
    """Check if a file is a video based on its MIME type."""
    mime, _ = mimetypes.guess_type(file_path)
    return mime is not None and mime.startswith("video")


def extract_frames(video_path, num_frames=8):
    """Extract frames from a video file at different parts."""
    print(f"🎬 Extracting {num_frames} frames from: {os.path.basename(video_path)}")
    
    # Create a temporary directory for this video's frames
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(TEMP_DIR, video_name)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Load the video
    clip = VideoFileClip(video_path)
    
    # Calculate frame positions (distributed across the video)
    duration = clip.duration
    
    # Generate evenly distributed frame times
    # Avoid the very beginning and end (first and last 5% of the video)
    start_percent = 0.05
    end_percent = 0.95
    usable_duration = end_percent - start_percent
    
    if num_frames == 1:
        # Just use the middle frame
        frame_times = [duration * 0.5]
    else:
        # Calculate evenly spaced intervals
        step = usable_duration / (num_frames - 1)
        frame_times = [duration * (start_percent + step * i) for i in range(num_frames)]
    
    frame_paths = []
    for i, time in enumerate(frame_times):
        # Get the frame at the specified time
        frame = clip.get_frame(time)
        
        # Convert to PIL Image and save
        img = Image.fromarray(frame)
        frame_path = os.path.join(frames_dir, f"frame_{i+1:02d}.jpg")
        img.save(frame_path, quality=95)
        frame_paths.append(frame_path)
        print(f"  📸 Extracted frame at {time:.2f}s: {os.path.basename(frame_path)}")
    
    # Close the video clip
    clip.close()
    
    return frame_paths, frames_dir


def get_frame_descriptions(model, frame_paths):
    """Get descriptions for each frame using the LLM."""
    descriptions = []
    
    for i, frame_path in enumerate(frame_paths, start=1):
        print(f"  🔍 Analyzing frame {i}/{len(frame_paths)}: {os.path.basename(frame_path)}")
        
        try:
            # Prepare the image for the LLM
            image_handle = lms.prepare_image(frame_path)
            chat = lms.Chat()
            
            # Create the prompt for the LLM
            prompt = """
                Describe the visual contents of this image with **objective, descriptive detail**. Focus on what is actually visible and how it would likely feel to a viewer. Include:

                - The main subject or environment (e.g. forest trail, empty hallway, abandoned playground)
                - The **atmosphere or emotional tone**, even if it's unsettling or eerie (e.g. cheerful, tense, creepy, surreal, depressing, dreamlike)
                - Color and lighting conditions (e.g. warm sunlight, cold grey tones, low visibility)
                - Weather, time of day, or visual effects if relevant

                Write **1 complete sentence**. Use neutral language — no exaggeration, no assumptions about beauty or meaning. **Do not soften or reframe dark or disturbing content**. Avoid metaphor, emojis, and opinions. This will be used to describe background visuals for file naming, so be precise and literal.
                """
            
            # Get the description from the LLM
            chat.add_user_message(prompt, images=[image_handle])
            prediction = model.respond(chat, response_format=FrameDescription)
            
            # Extract the description
            description = prediction.parsed["description"].strip()
            descriptions.append(description)
            print(f"  ✅ Got description for frame {i}")
            
        except Exception as e:
            print(f"  ❌ Error analyzing frame {i}: {str(e)}")
            descriptions.append(f"Error: {str(e)}")
    
    return descriptions


def generate_video_name(model, descriptions, video_path, min_length=32, max_length=128):
    """Generate a summary name for the video based on frame descriptions."""
    print(f"  📝 Generating name for video based on {len(descriptions)} frame descriptions")
    
    try:
        chat = lms.Chat()
        
        # Create the prompt for the LLM
        prompt = f"""
            You are helping organize a video collection. I have extracted {len(descriptions)} frames from a video and received these descriptions:

            {chr(10).join([f"Frame {i+1}: {desc}" for i, desc in enumerate(descriptions)])}

            Generate a **precise and informative filename** that summarizes the visual mood and content of this video based on those frames.

            Requirements:
            1. Use only **lowercase letters, numbers, and underscores**.
            2. Describe **what the video looks like** and **how it feels** to watch.
            3. Include important visible elements, mood, colors, and setting (e.g. gloomy_forest_with_mist, sunrise_over_pink_desert_road).
            4. NO file extension. NO special characters.
            5. The name must be **between {min_length} and {max_length} characters**.
            6. Avoid vague terms like "scenery", "footage", "image", "background".
            7. Do NOT use personal names or copyrighted terms.

            Goal: Help someone understand the look and emotional tone of the video at a glance from the filename alone.
            """
        
        # Get the summary from the LLM
        chat.add_user_message(prompt)
        prediction = model.respond(chat, response_format=VideoSummary)
        
        # Extract the suggested filename
        suggested_filename = prediction.parsed["suggested_filename"].strip()
        
        # Ensure the filename is within the specified length constraints
        if len(suggested_filename) < min_length:
            suggested_filename = suggested_filename.ljust(min_length, "_")
        if len(suggested_filename) > max_length:
            suggested_filename = suggested_filename[:max_length]
        
        # Ensure filename only contains safe characters
        suggested_filename = ''.join(c for c in suggested_filename if c.isalnum() or c == '_')
        
        print(f"  ✅ Generated name: {suggested_filename}")
        return suggested_filename
        
    except Exception as e:
        print(f"  ❌ Error generating video name: {str(e)}")
        # Fallback to a simple name based on the original filename
        original_name = os.path.splitext(os.path.basename(video_path))[0]
        return f"unnamed_{original_name}"


def save_metadata(video_path, descriptions, suggested_name):
    """Save metadata about the video analysis to a YAML file."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    metadata_dir = os.path.join(TEMP_DIR, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    metadata = {
        "original_filename": os.path.basename(video_path),
        "suggested_filename": suggested_name + os.path.splitext(video_path)[1],
        "frame_descriptions": [{"frame": i+1, "description": desc} for i, desc in enumerate(descriptions)]
    }
    
    metadata_path = os.path.join(metadata_dir, f"{video_name}.yaml")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"  💾 Saved metadata to: {os.path.basename(metadata_path)}")
    return metadata_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process videos in lib/videos folder using LLM for naming.")
    parser.add_argument("-m", "--model", default="gemma-3-4b-it", help="Model name to use for image analysis (default: gemma-3-4b-it)")
    parser.add_argument("-f", "--frames", type=int, default=8, help="Number of frames to extract per video (default: 8)")
    parser.add_argument("--min", "--min-length", type=int, default=32, help="Minimum filename length (default: 32)")
    parser.add_argument("--max", "--max-length", type=int, default=128, help="Maximum filename length (default: 128)")
    args = parser.parse_args()
    
    # Check if the videos directory exists
    if not os.path.exists(VIDEOS_DIR):
        print(f"❌ Error: Videos directory not found: {VIDEOS_DIR}")
        print(f"💡 Hint: Create the directory at {VIDEOS_DIR} and add video files to it.")
        sys.exit(1)
    
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Initialize the LLM
    try:
        model = lms.llm(args.model)
        print(f"🤖 Using model: {args.model}")
    except Exception as e:
        print(f"❌ Error initializing model: {str(e)}")
        sys.exit(1)
    
    # Find all video files in the videos directory
    video_files = [f for f in Path(VIDEOS_DIR).glob("*") if f.is_file() and is_video_file(f)]
    total_videos = len(video_files)
    
    if not video_files:
        print(f"⚠️ No video files found in {VIDEOS_DIR}")
        sys.exit(0)
    
    print(f"🎥 Found {total_videos} video(s). Starting processing...\n")
    
    # Process each video sequentially
    for idx, video_path in enumerate(video_files, start=1):
        print(f"\n[{idx}/{total_videos}] Processing video: {video_path.name}")
        
        # Create a temporary directory for this video's frames
        video_name = os.path.splitext(video_path.name)[0]
        frames_dir = os.path.join(TEMP_DIR, video_name)
        os.makedirs(frames_dir, exist_ok=True)
        
        frame_paths = []
        
        try:
            # Extract frames from the video
            frame_paths, frames_dir = extract_frames(str(video_path), args.frames)
            
            # Get descriptions for each frame
            descriptions = get_frame_descriptions(model, frame_paths)
            
            # Generate a summary name for the video
            suggested_name = generate_video_name(model, descriptions, str(video_path), 
                                                args.min, args.max)
            
            # Save metadata
            metadata_path = save_metadata(str(video_path), descriptions, suggested_name)
            
            # Rename the video file
            new_filename = suggested_name + os.path.splitext(str(video_path))[1]
            new_filepath = os.path.join(VIDEOS_DIR, new_filename)
            
            if os.path.exists(new_filepath):
                print(f"⏭️ Skipped renaming: {new_filename} already exists.")
            else:
                # Rename using os functions to ensure it works with string paths
                try:
                    os.rename(str(video_path), new_filepath)
                    print(f"✅ Renamed video to: {new_filename}")
                except Exception as e:
                    print(f"❌ Error renaming file: {str(e)}")
            
            print(f"✅ [{idx}/{total_videos}] Completed processing: {video_path.name}")
            
        except Exception as e:
            print(f"❌ [{idx}/{total_videos}] Error processing {video_path.name}: {str(e)}")
        
        finally:
            # Clean up temporary files regardless of success or failure
            print(f"  🧹 Cleaning up temporary files...")
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    try:
                        os.remove(frame_path)
                    except Exception:
                        pass
            
            if os.path.exists(frames_dir):
                try:
                    shutil.rmtree(frames_dir)
                except Exception:
                    pass
    
    # Clean up the entire temp directory at the end
    try:
        if os.path.exists(TEMP_DIR):
            print(f"\n🧹 Cleaning up temp directory...")
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR, exist_ok=True)  # Recreate an empty temp directory
    except Exception as e:
        print(f"⚠️ Warning: Could not clean up temp directory: {str(e)}")
    
    print("\n🎉 All done!")
    
    # Unload the model to free up resources
    model.unload()

if __name__ == "__main__":
    main()
