#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Master script for DeepVideo2 that sequentially runs all steps of the video generation process:
1. make_scenarios.py - Generate scenario scripts
2. make_voice_lines.py - Generate voice lines for each scenario
3. make_videos.py - Generate videos from scenarios and voice lines
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
    parser = argparse.ArgumentParser(description="DeepVideo2 Master Script - Generate complete videos in one step")
    
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    parser.add_argument("-n", "--num", type=int, default=1, help="Target number of scenarios/videos to have (default: 1)")
    parser.add_argument("-q", "--quality", type=float, default=1.0, help="Video quality factor (0.0-1.0, default: 1.0)")
    parser.add_argument("-m", "--model", type=str, help="Custom LLM model to use (overrides config)")
    parser.add_argument("--skip-voices", action="store_true", help="Skip voice generation step")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
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

def count_existing_scenarios(project_name):
    """Count the number of existing scenario files for the project."""
    scenarios_dir = os.path.join("output", project_name, "scenarios")
    
    # Create directory if it doesn't exist
    if not os.path.exists(scenarios_dir):
        os.makedirs(scenarios_dir, exist_ok=True)
        return 0
    
    # Count YAML files in the scenarios directory
    scenario_files = glob.glob(os.path.join(scenarios_dir, "*.yaml"))
    return len(scenario_files)

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
    
    # Validate config file
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        print(f"💡 Hint: Make sure the config file exists. Example: configs/sample.yaml")
        return 1
    
    # Validate quality
    if args.quality <= 0 or args.quality > 1:
        print(f"❌ Error: Quality must be between 0.0 and 1.0")
        return 1
    
    # Load config to get project name
    config = load_config(args.config)
    project_name = config.get("project_name", "default")
    
    # Count existing scenarios
    existing_count = count_existing_scenarios(project_name)
    
    print(f"\n{'='*70}")
    print(f"🎬 DEEPVIDEO2 MASTER SCRIPT")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    print(f"Target number of videos: {args.num}")
    print(f"Existing scenarios: {existing_count}")
    print(f"Quality factor: {args.quality}")
    if args.model:
        print(f"Using custom model: {args.model}")
    if args.skip_voices:
        print(f"Voice generation: SKIPPED")
    print(f"{'='*70}\n")
    
    # Step 1: Generate scenarios (only if needed)
    scenarios_to_generate = max(0, args.num - existing_count)
    
    if scenarios_to_generate > 0:
        print(f"Generating {scenarios_to_generate} new scenarios to reach target of {args.num}")
        scenario_args = ["-c", args.config, "-n", str(scenarios_to_generate)]
        # Add model parameter if specified
        if args.model:
            scenario_args.extend(["-m", args.model])
        exit_code = run_script("make_scenarios.py", scenario_args)
        if exit_code != 0:
            print("❌ Failed to generate scenarios. Stopping.")
            return exit_code
    else:
        print(f"✅ Already have {existing_count} scenarios, which meets or exceeds target of {args.num}. Skipping scenario generation.")
    
    # Step 2: Generate voice lines for all scenarios
    if not args.skip_voices:
        voice_args = ["-c", args.config]
        exit_code = run_script("make_voice_lines.py", voice_args)
        if exit_code != 0:
            print("❌ Failed to generate voice lines. Stopping.")
            return exit_code
    
    # Step 3: Generate videos
    # Use -1 to process all unprocessed scenarios
    video_args = ["-c", args.config, "-n", "-1"]
    if args.quality != 1.0:
        video_args.extend(["-q", str(args.quality)])
    if args.skip_voices:
        video_args.append("--skip-voices")
    exit_code = run_script("make_videos.py", video_args)
    if exit_code != 0:
        print("❌ Failed to generate videos. Stopping.")
        return exit_code
    
    print(f"\n{'='*70}")
    print(f"🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user (Ctrl+C)")
        print("🛑 Exiting gracefully...")
        sys.exit(130)  # Standard exit code for SIGINT
