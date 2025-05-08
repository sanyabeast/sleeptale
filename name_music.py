#!/usr/bin/env python3
"""
A utility script to rename music files in lib/music from:
'NAME - a fools theme, GENRE - cinematic, MOOD - calm, ARTIST - audio hertz.mp3'
to:
'NAME_a_fools_theme_GENRE_cinematic_MOOD_calm_ARTIST_audio_hertz.mp3'

Format rules:
- Labels (NAME, GENRE, MOOD, ARTIST) are uppercase
- Values are lowercase
- Spaces in values are preserved for ARTIST, converted to underscores for others
- Punctuation is removed
"""

import os
import re
import sys
from pathlib import Path

# Get the absolute path of the project directory
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
MUSIC_DIR = os.path.join(PROJECT_DIR, "lib", "music")

def sanitize_to_snake_case(value):
    """Convert text to snake_case (lowercase with underscores)."""
    # Remove punctuation, convert to lowercase
    value = re.sub(r'[^\w\s]', '', value.lower())
    # Replace spaces with underscores
    value = re.sub(r'\s+', '_', value)
    # Remove leading/trailing underscores
    value = re.sub(r'^_+|_+$', '', value)
    return value

def sanitize_artist(value):
    """Sanitize artist name and convert spaces to underscores."""
    # Remove punctuation, convert to lowercase
    value = re.sub(r'[^\w\s]', '', value.lower())
    # Replace spaces with underscores
    value = re.sub(r'\s+', '_', value)
    # Remove leading/trailing underscores
    value = re.sub(r'^_+|_+$', '', value)
    return value

def rename_music_files():
    """Rename music files in the lib/music directory."""
    # Check if the music directory exists
    if not os.path.exists(MUSIC_DIR):
        print(f"❌ Error: Music directory not found: {MUSIC_DIR}")
        print(f"💡 Hint: Create the directory at {MUSIC_DIR} and add music files to it.")
        return False
    
    # Get all files in the music directory
    music_files = [f for f in os.listdir(MUSIC_DIR) if os.path.isfile(os.path.join(MUSIC_DIR, f))]
    
    if not music_files:
        print(f"⚠️ No music files found in {MUSIC_DIR}")
        return False
    
    print(f"🎵 Found {len(music_files)} music file(s). Starting renaming...\n")
    
    renamed_count = 0
    skipped_count = 0
    already_formatted_count = 0
    
    for filename in music_files:
        file_ext = os.path.splitext(filename)[1]
        
        # Check if the file is already in the correct format
        if (filename.startswith("NAME_") and 
            "_GENRE_" in filename and 
            "_MOOD_" in filename and 
            "_ARTIST_" in filename):
            
            # Check if there are spaces in the ARTIST part
            if " " in filename:
                # Extract the components
                parts = filename.split("_ARTIST_")
                if len(parts) == 2:
                    prefix = parts[0] + "_ARTIST_"
                    artist_part = parts[1]
                    
                    # Split artist part by extension
                    artist_name, ext = os.path.splitext(artist_part)
                    
                    # Replace spaces with underscores in artist name
                    artist_name_fixed = artist_name.replace(" ", "_")
                    
                    # Create new filename
                    new_filename = prefix + artist_name_fixed + ext
                    
                    # Rename the file
                    old_path = os.path.join(MUSIC_DIR, filename)
                    new_path = os.path.join(MUSIC_DIR, new_filename)
                    
                    try:
                        os.rename(old_path, new_path)
                        print(f"✅ Fixed spaces in artist: {filename} → {new_filename}")
                        renamed_count += 1
                    except Exception as e:
                        print(f"❌ Error renaming {filename}: {str(e)}")
                        skipped_count += 1
                else:
                    print(f"⏭️ Skipping {filename} (already in correct format)")
                    already_formatted_count += 1
            else:
                print(f"⏭️ Skipping {filename} (already in correct format)")
                already_formatted_count += 1
            continue
        
        # Skip files that don't match the expected original format
        if not (", GENRE -" in filename and ", MOOD -" in filename and ", ARTIST -" in filename):
            print(f"⏭️ Skipping {filename} (doesn't match expected format)")
            skipped_count += 1
            continue
        
        # Extract components using regex pattern matching
        # This pattern looks for "NAME - value, LABEL - value" patterns
        components = {}
        
        # First, handle the NAME part which is at the beginning and has a different format
        name_match = re.match(r"NAME - ([^,]+),", filename)
        if name_match:
            name_value = name_match.group(1).strip()
            components["NAME"] = sanitize_to_snake_case(name_value)
        
        # Handle GENRE and MOOD with snake_case
        for label in ["GENRE", "MOOD"]:
            pattern = rf"{label} - ([^,]+)(?:,|{re.escape(file_ext)})"
            match = re.search(pattern, filename)
            if match:
                value = match.group(1).strip()
                components[label] = sanitize_to_snake_case(value)
        
        # Handle ARTIST preserving spaces
        artist_pattern = r"ARTIST - ([^,]+)(?:,|" + re.escape(file_ext) + ")"
        artist_match = re.search(artist_pattern, filename)
        if artist_match:
            artist_value = artist_match.group(1).strip()
            components["ARTIST"] = sanitize_artist(artist_value)
        
        # Check if we have all components
        if len(components) < 4:
            print(f"⏭️ Skipping {filename} (couldn't extract all components)")
            skipped_count += 1
            continue
        
        # Create new filename
        new_filename = (
            f"NAME_{components['NAME']}_"
            f"GENRE_{components['GENRE']}_"
            f"MOOD_{components['MOOD']}_"
            f"ARTIST_{components['ARTIST']}{file_ext}"
        )
        
        # Rename the file
        old_path = os.path.join(MUSIC_DIR, filename)
        new_path = os.path.join(MUSIC_DIR, new_filename)
        
        # Check if the new filename already exists
        if os.path.exists(new_path):
            print(f"⚠️ Skipping {filename} (new filename already exists: {new_filename})")
            skipped_count += 1
            continue
        
        try:
            os.rename(old_path, new_path)
            print(f"✅ Renamed: {filename} → {new_filename}")
            renamed_count += 1
        except Exception as e:
            print(f"❌ Error renaming {filename}: {str(e)}")
            skipped_count += 1
    
    print(f"\n🎉 Renaming complete! Renamed {renamed_count} files, skipped {skipped_count} files, already formatted {already_formatted_count} files.")
    return True

if __name__ == "__main__":
    print("🎵 Music File Renamer")
    print("=====================")
    print("This script will rename music files from:")
    print("'NAME - a fools theme, GENRE - cinematic, MOOD - calm, ARTIST - audio hertz.mp3'")
    print("to:")
    print("'NAME_a_fools_theme_GENRE_cinematic_MOOD_calm_ARTIST_audio_hertz.mp3'")
    print("=====================")
    
    # Ask for confirmation
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        confirm = "y"
    else:
        confirm = input("Do you want to continue? (y/n): ").strip().lower()
    
    if confirm in ["y", "yes"]:
        rename_music_files()
    else:
        print("Operation cancelled.")
