#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Master script for SleepTale that sequentially runs all steps of the video generation process:
1. make_story.py - Generate sleep stories
2. make_voice_lines.py - Generate voice lines for each story
3. make_video.py - Generate videos from stories and voice lines
"""

import argparse
import os
import sys
import subprocess
import time
import yaml
import glob

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SleepTale Master Script - Generate complete sleep videos in one step")
    
    parser.add_argument("-c", type=int, default=1, help="Number of stories to generate (default: 1)")
    parser.add_argument("-m", "--model", type=str, help="Model to use for story generation (overrides config)")
    parser.add_argument("-d", "--duration", type=int, help="Duration of stories in minutes (overrides config)")
    parser.add_argument("-q", "--quality", type=float, default=1.0, help="Quality factor for video resolution (1.0 = 1080p, 0.5 = 540p)")
    
    return parser.parse_args()

def load_config():
    """Load configuration from config.yaml file."""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print("❌ Error: config.yaml not found in the root directory")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config.yaml: {e}")
        sys.exit(1)
        sys.exit(1)

def count_existing_stories():
    """Count the number of existing story files."""
    stories_dir = os.path.join("output", "stories")
    
    # Create directory if it doesn't exist
    if not os.path.exists(stories_dir):
        os.makedirs(stories_dir, exist_ok=True)
        return 0
    
    # Count YAML files in the stories directory
    story_files = glob.glob(os.path.join(stories_dir, "*.yaml"))
    return len(story_files)

def run_script(script_name, args):
    """Run a script with the given arguments and return its exit code."""
    print(f"\n{'='*70}")
    print(f"🚀 Running {script_name}")
    print(f"{'='*70}")
    
    cmd = [sys.executable, script_name]
    cmd.extend(args)
    
    print(f"Command: {' '.join(cmd)}")
    print(f"{'-'*70}")
    
    start_time = time.time()
    try:
        process = subprocess.run(cmd)
        elapsed_time = time.time() - start_time
        
        print(f"{'-'*70}")
        print(f"✅ Finished {script_name} (Exit code: {process.returncode}, Time: {elapsed_time:.2f}s)")
        
        return process.returncode
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print(f"\n{'-'*70}")
        print(f"⚠️ {script_name} interrupted by user after {elapsed_time:.2f}s")
        return 130  # Standard exit code for SIGINT (Ctrl+C)

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Validate quality
    if args.quality <= 0 or args.quality > 1:
        print(f"❌ Error: Quality must be between 0.0 and 1.0")
        return 1
    
    # Load config
    config = load_config()
    project_name = config.get("project_name", "default")
    
    print(f"🌟 Starting SleepTale master script")
    
    # Step 1: Generate stories
    existing_stories = count_existing_stories()
    stories_to_generate = max(0, args.c - existing_stories)
    
    if stories_to_generate > 0:
        print(f"📝 Generating {stories_to_generate} new stories")
        
        # Generate each story
        for i in range(args.c):
            print(f"\n📝 Generating story {i+1} of {stories_to_generate}")
            
            # Build arguments for make_story.py
            story_args = []
            
            if args.model:
                story_args.extend(["-m", args.model])
            
            if args.duration:
                story_args.extend(["-d", str(args.duration)])
            
            # Run the story generation script
            exit_code = run_script("make_story.py", story_args)
            if exit_code != 0:
                print(f"❌ Story generation failed with exit code {exit_code}")
                return exit_code
    else:
        print(f"✅ Already have {existing_stories} stories, no need to generate more")
    
    # Step 2: Generate voice lines
    voice_args = []
    
    exit_code = run_script("make_voice_lines.py", voice_args)
    if exit_code != 0:
        print(f"❌ Voice line generation failed with exit code {exit_code}")
        return exit_code
    
    # Step 3: Generate videos
    video_args = []
    
    if args.quality:
        video_args.extend(["-q", str(args.quality)])
    
    exit_code = run_script("make_video.py", video_args)
    if exit_code != 0:
        print(f"❌ Video generation failed with exit code {exit_code}")
        return exit_code
    
    print("\n" + "="*70)
    print(f"🎉 All steps completed successfully!")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user (Ctrl+C)")
        print("🛑 Exiting gracefully...")
        sys.exit(130)  # Standard exit code for SIGINT
