#!/usr/bin/env python
"""
Clean script for DeepVideo2 project.

This script allows you to clean different types of generated content:
- videos: Delete generated video files
- voices: Delete generated voice line files
- images: Delete generated image files
- scenarios: Delete generated scenario files
- all: Delete all generated content for the project

You can also use the --all-projects flag to clean all output directories without requiring a config file.
"""

import os
import yaml
import argparse
import shutil
import sys
import glob

# Get the absolute path of the project directory
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        print("❌ Error: No config file specified.")
        print("💡 Hint: Use -c or --config to specify a config file. Example: -c configs/sample.yaml")
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
            print(f"ℹ️ Using config filename '{config_name}' as project name")
        
        return config
    except FileNotFoundError:
        print(f"❌ Error: Config file not found: {config_path}")
        print(f"💡 Hint: Make sure the config file exists. Example: configs/sample.yaml")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config file: {e}")
        sys.exit(1)

def delete_videos(config, dry_run=False):
    """Delete all generated video files for the project."""
    project_name = config.get("project_name")
    
    print(f"\n🎬 Cleaning videos for project: {project_name}")
    
    # Get the videos directory
    output_dir = os.path.join(PROJECT_DIR, "output", project_name)
    videos_dir = os.path.join(output_dir, config.get("directories", {}).get("output_videos", "videos"))
    
    # Check if the directory exists
    if not os.path.exists(videos_dir):
        print(f"⚠️ Videos directory not found: {videos_dir}")
        return
    
    # Find all MP4 files in the videos directory
    video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
    
    if not video_files:
        print(f"⚠️ No video files found in {videos_dir}")
        return
    
    # Track how many files were deleted
    deleted_count = 0
    total_count = len(video_files)
    
    for filepath in video_files:
        filename = os.path.basename(filepath)
        print(f"🗑️ Deleting: {filename}")
        
        if not dry_run:
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error deleting {filename}: {str(e)}")
    
    if not dry_run:
        print(f"✅ Deleted {deleted_count} of {total_count} video files")
    else:
        print(f"🔍 Dry run: Would delete {total_count} video files")

def delete_voice_lines(config, dry_run=False):
    """Delete all generated voice line files for the project."""
    project_name = config.get("project_name")
    
    print(f"\n🎤 Cleaning voice lines for project: {project_name}")
    
    # Get the voice lines directory
    output_dir = os.path.join(PROJECT_DIR, "output", project_name)
    voice_lines_dir = os.path.join(output_dir, config.get("directories", {}).get("voice_lines", "voice_lines"))
    
    # Check if the directory exists
    if not os.path.exists(voice_lines_dir):
        print(f"⚠️ Voice lines directory not found: {voice_lines_dir}")
        return
    
    # Find all audio files in the voice lines directory
    audio_files = glob.glob(os.path.join(voice_lines_dir, "*.mp3")) + glob.glob(os.path.join(voice_lines_dir, "*.wav"))
    
    if not audio_files:
        print(f"⚠️ No voice line files found in {voice_lines_dir}")
        return
    
    # Track how many files were deleted
    deleted_count = 0
    total_count = len(audio_files)
    
    for filepath in audio_files:
        filename = os.path.basename(filepath)
        print(f"🗑️ Deleting: {filename}")
        
        if not dry_run:
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error deleting {filename}: {str(e)}")
    
    if not dry_run:
        print(f"✅ Deleted {deleted_count} of {total_count} voice line files")
    else:
        print(f"🔍 Dry run: Would delete {total_count} voice line files")

def delete_images(config, dry_run=False):
    """Delete all generated image files for the project."""
    project_name = config.get("project_name")
    
    print(f"\n🖼️ Cleaning images for project: {project_name}")
    
    # Get the images directory
    output_dir = os.path.join(PROJECT_DIR, "output", project_name)
    images_dir = os.path.join(output_dir, config.get("directories", {}).get("images", "images"))
    
    # Check if the directory exists
    if not os.path.exists(images_dir):
        print(f"⚠️ Images directory not found: {images_dir}")
        return
    
    # Find all image files in the images directory
    image_files = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
    
    if not image_files:
        print(f"⚠️ No image files found in {images_dir}")
        return
    
    # Track how many files were deleted
    deleted_count = 0
    total_count = len(image_files)
    
    for filepath in image_files:
        filename = os.path.basename(filepath)
        print(f"🗑️ Deleting: {filename}")
        
        if not dry_run:
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error deleting {filename}: {str(e)}")
    
    if not dry_run:
        print(f"✅ Deleted {deleted_count} of {total_count} image files")
    else:
        print(f"🔍 Dry run: Would delete {total_count} image files")

def delete_scenarios(config, dry_run=False):
    """Delete all generated scenario files for the project."""
    project_name = config.get("project_name")
    
    print(f"\n📝 Cleaning scenarios for project: {project_name}")
    
    # Get the scenarios directory
    output_dir = os.path.join(PROJECT_DIR, "output", project_name)
    scenarios_dir = os.path.join(output_dir, config.get("directories", {}).get("scenarios", "scenarios"))
    
    # Check if the directory exists
    if not os.path.exists(scenarios_dir):
        print(f"⚠️ Scenarios directory not found: {scenarios_dir}")
        return
    
    # Find all YAML files in the scenarios directory
    scenario_files = glob.glob(os.path.join(scenarios_dir, "*.yaml"))
    
    if not scenario_files:
        print(f"⚠️ No scenario files found in {scenarios_dir}")
        return
    
    # Track how many files were deleted
    deleted_count = 0
    total_count = len(scenario_files)
    
    for filepath in scenario_files:
        filename = os.path.basename(filepath)
        print(f"🗑️ Deleting: {filename}")
        
        if not dry_run:
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error deleting {filename}: {str(e)}")
    
    if not dry_run:
        print(f"✅ Deleted {deleted_count} of {total_count} scenario files")
    else:
        print(f"🔍 Dry run: Would delete {total_count} scenario files")

def clean_all_project_content(config, dry_run=False):
    """Clean all content for a specific project."""
    project_name = config.get("project_name")
    
    print(f"\n🧹 Cleaning all content for project: {project_name}")
    
    # Clean each type of content
    delete_videos(config, dry_run)
    delete_voice_lines(config, dry_run)
    delete_images(config, dry_run)
    delete_scenarios(config, dry_run)
    
    print(f"\n✅ Finished cleaning all content for project: {project_name}")

def clean_all_projects(dry_run=False):
    """Remove the entire output directory and recreate it empty."""
    output_dir = os.path.join(PROJECT_DIR, "output")
    
    if not os.path.exists(output_dir):
        print(f"⚠️ Output directory not found: {output_dir}")
        return
    
    print(f"🗑️ Deleting all output directories: {output_dir}")
    
    if not dry_run:
        try:
            # Remove the entire output directory
            shutil.rmtree(output_dir)
            
            # Recreate the output directory
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"✅ Deleted and recreated output directory: {output_dir}")
        except Exception as e:
            print(f"❌ Error cleaning output directory: {str(e)}")
    else:
        print(f"🔍 Dry run: Would delete and recreate output directory: {output_dir}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean generated content for DeepVideo2 projects")
    parser.add_argument("-c", "--config", help="Path to configuration file")
    parser.add_argument("--videos", action="store_true", help="Clean generated video files")
    parser.add_argument("--voices", action="store_true", help="Clean generated voice line files")
    parser.add_argument("--images", action="store_true", help="Clean generated image files")
    parser.add_argument("--scenarios", action="store_true", help="Clean generated scenario files")
    parser.add_argument("--all", action="store_true", help="Clean all generated content for the project")
    parser.add_argument("--all-projects", action="store_true", help="Clean all output directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    print("\n" + "="*50)
    print("🧹 DeepVideo2 Clean Tool")
    print("="*50)
    
    # Check if we should clean all projects
    if args.all_projects:
        print("\n🔄 Cleaning all projects")
        if args.dry_run:
            print("🔍 DRY RUN: No files will be deleted")
        
        clean_all_projects(args.dry_run)
    else:
        # Require config file if not using --all-projects
        if not args.config:
            print("❌ Error: No config file specified.")
            print("💡 Hint: Use -c or --config to specify a config file, or use --all-projects to clean all output.")
            return 1
        
        # Load configuration
        config = load_config(args.config)
        
        # Check which cleaning operations to perform
        if args.all:
            clean_all_project_content(config, args.dry_run)
        else:
            # Check if at least one cleaning option was specified
            if not (args.videos or args.voices or args.images or args.scenarios):
                print("❌ Error: No cleaning option specified.")
                print("💡 Hint: Use --videos, --voices, --images, --scenarios, or --all to specify what to clean.")
                return 1
            
            # Perform the requested cleaning operations
            if args.videos:
                delete_videos(config, args.dry_run)
            
            if args.voices:
                delete_voice_lines(config, args.dry_run)
            
            if args.images:
                delete_images(config, args.dry_run)
            
            if args.scenarios:
                delete_scenarios(config, args.dry_run)
    
    print("\n" + "="*50)
    print("🎉 Clean operation completed")
    print("="*50 + "\n")
    
    return 0

if __name__ == "__main__":
    main()
