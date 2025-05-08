#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SleepTeller - Story Generation Module

This script generates long, monotonous stories using an LLM (Large Language Model)
by iteratively feeding the last part of the output into new prompts until the
desired length is reached.
"""

import os
import sys
import yaml
import argparse
import time
import random
import json
import requests
from pathlib import Path

# Get the absolute path of the project directory
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

# ─────────────────────────────────────────────────────
# 🎲 CONFIGURATION
# ─────────────────────────────────────────────────────
def log(message, emoji=None, indent=0):
    """Standardized logging function with consistent emoji spacing.
    
    Args:
        message: The message to log
        emoji: Optional emoji to prefix the message
        indent: Number of spaces to indent the message
    """
    indent_str = ' ' * indent
    if emoji:
        # Ensure there's a space after the emoji
        print(f"{indent_str}{emoji} {message}")
    else:
        print(f"{indent_str}{message}")

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
PROJECT_NAME = None

def update_directories():
    """Update directory paths based on the loaded configuration."""
    global STORIES_DIR, PROJECT_NAME
    
    PROJECT_NAME = CONFIG.get('project_name', 'default')
    
    # Create output directory structure
    output_dir = os.path.join(PROJECT_DIR, "output", PROJECT_NAME)
    STORIES_DIR = os.path.join(output_dir, "stories")
    
    # Create directories if they don't exist
    os.makedirs(STORIES_DIR, exist_ok=True)
    
    log(f"Project: {PROJECT_NAME}", "📂")
    log(f"Stories directory: {STORIES_DIR}", "📝")

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

# ─────────────────────────────────────────────────────
# 🧠 LLM INTERACTION
# ─────────────────────────────────────────────────────
def initialize_model(model_name=None, seed=None):
    """Initialize the LLM model parameters.
    
    Args:
        model_name: Name of the model to use
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with model parameters
    """
    if model_name is None:
        model_name = CONFIG.get('story', {}).get('model', 'gemma-3-4b-it-qat')
    
    log(f"Using model: {model_name}", "🤖")
    
    # Initialize with a fixed seed if provided
    seed_value = seed if seed is not None else random.randint(0, 1000000)
    if seed is not None:
        log(f"Using fixed seed: {seed_value}", "🌱")
    
    # Return model parameters as a dictionary
    return {
        "model": model_name,
        "seed": seed_value,
        "api_base": CONFIG.get('llm', {}).get('api_base', 'http://localhost:1234/v1')
    }

def query_llm(model_params, prompt, max_tokens=None, temperature=0.7):
    """Query the LLM using direct API calls.
    
    Args:
        model_params: Dictionary with model parameters
        prompt: The prompt to send to the LLM
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature parameter for generation
    
    Returns:
        Generated text from the LLM
    """
    if max_tokens is None:
        max_tokens = CONFIG.get('story', {}).get('max_tokens_per_chunk', 1000)
    
    # Include the storyteller instructions in the prompt itself
    full_prompt = """You are a storyteller who specializes in creating long, soothing, monotonous stories to help people fall asleep. Your stories should be calming, low-stimulation, and avoid exciting plot twists or emotional highs.

{}""".format(prompt)
    
    # Prepare the request
    headers = {
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': model_params['model'],
        'messages': [
            {'role': 'user', 'content': full_prompt}
        ],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'seed': model_params['seed']
        # Remove response_format as it's not supported by this API version
    }
    
    api_base = model_params['api_base']
    log(f"API endpoint: {api_base}", "🔌")
    
    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=data,
            timeout=120  # 2-minute timeout
        )
        
        if response.status_code != 200:
            log(f"Error from LLM API: {response.status_code} - {response.text}", "❌")
            return None
        
        result = response.json()
        response_text = result['choices'][0]['message']['content']
        
        # Try to parse as JSON in case the model returned JSON format anyway
        # (some models might still return JSON even if not requested)
        try:
            parsed_response = json.loads(response_text)
            if 'content' in parsed_response:
                return parsed_response['content']
            else:
                # If no 'content' field, return the raw text
                return response_text
        except json.JSONDecodeError:
            # If it's not JSON, just return the raw text
            return response_text
    
    except requests.exceptions.RequestException as e:
        log(f"Error connecting to LLM API: {e}", "❌")
        log("Hint: Make sure LM Studio is running and the API is enabled.", "💡")
        return None

def estimate_story_length(text, target_minutes):
    """Estimate if the story has reached the target length.
    
    Args:
        text: The current story text
        target_minutes: Target duration in minutes
    
    Returns:
        Boolean indicating if the target length has been reached
    """
    # Average reading speed is about 150-200 words per minute for narration
    # We'll use a conservative estimate of 150 words per minute
    words = len(text.split())
    estimated_minutes = words / 150
    
    return estimated_minutes >= target_minutes

def generate_story(topic, target_duration_minutes, model_name=None, max_tokens_per_chunk=None, temperature=0.7, seed=None, chunk_size=None, context_size=None, max_chunks_in_memory=3):
    """Generate a long story by iteratively querying the LLM.
    
    Args:
        topic: The topic or theme for the story
        target_duration_minutes: Target duration in minutes
        model_name: Name of the model to use
        max_tokens_per_chunk: Maximum tokens per LLM request
        temperature: Temperature parameter for generation
        seed: Random seed for reproducibility
        chunk_size: Number of words per chunk (default: 500)
        context_size: Number of words to keep as context from previous chunks (default: 200)
    
    Returns:
        The generated story text
    """
    # Set defaults for chunk and context size
    if chunk_size is None:
        chunk_size = CONFIG.get('story', {}).get('chunk_size', 500)
    if context_size is None:
        context_size = CONFIG.get('story', {}).get('context_size', 200)
        
    # Create a temporary file to store chunks as we generate them
    temp_story_file = os.path.join(STORIES_DIR, f"temp_{int(time.time())}.txt")
    
    # Track total word count and duration
    total_word_count = 0
    total_minutes = 0
    
    log(f"Generating story on topic: '{topic}'", "🧠")
    log(f"Target duration: {target_duration_minutes} minutes", "⏱️")
    log(f"Chunk size: ~{chunk_size} words", "📏")
    log(f"Context size: ~{context_size} words", "🔄")
    
    if seed is not None:
        log(f"Using seed: {seed}", "🌱")
    
    # Initialize model parameters
    model_params = initialize_model(model_name, seed)
    
    # Initial prompt to start the story
    initial_prompt = f"""Write a long, soothing, monotonous story about {topic}. 
    The story should be calming and help people fall asleep. 
    Avoid exciting plot twists or emotional highs. 
    Use a slow, gentle pace with lots of peaceful descriptions.
    Write approximately {chunk_size} words.
    Start the story now with a title:"""
    
    log("Generating initial story chunk...", "✍️")
    current_chunk = query_llm(model_params, initial_prompt, max_tokens_per_chunk, temperature)
    
    if current_chunk is None:
        log("Failed to generate initial story chunk", "❌")
        return None
    
    # Track chunk count and total words
    chunk_count = 1
    chunk_words = len(current_chunk.split())
    total_word_count += chunk_words
    total_minutes = total_word_count / 150  # Assuming 150 words per minute
    
    # Write the first chunk to the temp file
    with open(temp_story_file, 'w', encoding='utf-8') as f:
        f.write(current_chunk)
    
    log(f"Chunk {chunk_count}/{target_duration_minutes * 150 // chunk_size} (est.): {chunk_words} words", "📝")
    
    # Show a preview of the generated text
    preview_words = min(20, len(current_chunk.split()))
    preview_text = " ".join(current_chunk.split()[:preview_words]) + "..."
    log(f"Preview: {preview_text}", "🔎", indent=2)
    
    # Keep track of recent chunks for context
    recent_chunks = [current_chunk]
    
    # Keep generating more content until we reach the target duration
    while total_minutes < target_duration_minutes:
        # Get context from recent chunks
        combined_recent = "\n\n".join(recent_chunks)
        words = combined_recent.split()
        context_words = words[-min(context_size, len(words)):]
        context = " ".join(context_words)
        
        continuation_prompt = f"""Continue the following story in the same slow, soothing style. 
        Pick up exactly where it left off, maintaining the same peaceful atmosphere.
        Write approximately {chunk_size} more words.
        
        Previous text ending:
        {context}
        """
        
        chunk_count += 1
        log(f"Generating chunk {chunk_count}/{target_duration_minutes * 150 // chunk_size} (est.)...", "✍️")
        next_chunk = query_llm(model_params, continuation_prompt, max_tokens_per_chunk, temperature)
        
        if next_chunk is None:
            log(f"Failed to generate chunk {chunk_count}", "❌")
            break
        
        # Add to recent chunks and maintain max chunks in memory
        recent_chunks.append(next_chunk)
        if len(recent_chunks) > max_chunks_in_memory:
            recent_chunks.pop(0)
        
        # Append the new content to the file
        with open(temp_story_file, 'a', encoding='utf-8') as f:
            f.write("\n\n" + next_chunk)
        
        # Calculate and show progress
        chunk_words = len(next_chunk.split())
        total_word_count += chunk_words
        total_minutes = total_word_count / 150  # Assuming 150 words per minute
        
        log(f"Chunk {chunk_count}/{target_duration_minutes * 150 // chunk_size} (est.): {chunk_words} words", "📝")
        log(f"Total: {total_word_count} words, ~{total_minutes:.1f} minutes / {target_duration_minutes} minutes ({(total_minutes/target_duration_minutes*100):.1f}%)", "📈")
        
        # Show a preview of the latest chunk
        preview_words = min(20, len(next_chunk.split()))
        preview_text = " ".join(next_chunk.split()[:preview_words]) + "..."
        log(f"Preview: {preview_text}", "🔎", indent=2)
    
    # No cleanup needed with direct API calls
    
    # Read the complete story from the temp file
    with open(temp_story_file, 'r', encoding='utf-8') as f:
        complete_story = f.read()
    
    # Delete the temp file
    try:
        os.remove(temp_story_file)
    except Exception as e:
        log(f"Warning: Failed to delete temp file: {e}", "⚠️")
    
    return complete_story

def save_story(story_text, topic):
    """Save the generated story to a file.
    
    Args:
        story_text: The generated story text
        topic: The topic of the story
    
    Returns:
        Path to the saved story file
    """
    # Create a filename based on the topic
    safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)
    safe_topic = safe_topic.replace(" ", "_").lower()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_topic}_{timestamp}.txt"
    
    file_path = os.path.join(STORIES_DIR, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(story_text)
    
    log(f"Story saved to: {file_path}", "💾")
    return file_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a long, monotonous story using an LLM")
    
    parser.add_argument("-c", "--config", default="configs/sample.yaml", help="Path to the config file")
    parser.add_argument("-t", "--topic", help="Override the story topic from config")
    parser.add_argument("-d", "--duration", type=int, help="Override the target duration in minutes")
    parser.add_argument("-m", "--model", help="Override the model to use")
    parser.add_argument("-s", "--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument("--chunk-size", type=int, help="Number of words per chunk (default: 500)")
    parser.add_argument("--context-size", type=int, help="Number of words to keep as context from previous chunks (default: 200)")
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    global CONFIG
    CONFIG = load_config(args.config)
    
    # Update directories
    update_directories()
    
    # Get story parameters from config
    topic = args.topic or CONFIG.get('story', {}).get('topic', 'a gentle walk through a quiet forest')
    target_duration = args.duration or CONFIG.get('story', {}).get('target_duration_minutes', 20)
    model_name = args.model or CONFIG.get('story', {}).get('model')
    max_tokens = CONFIG.get('story', {}).get('max_tokens_per_chunk')
    temperature = CONFIG.get('story', {}).get('temperature', 0.7)
    seed = args.seed or CONFIG.get('llm', {}).get('seed')
    chunk_size = args.chunk_size or CONFIG.get('story', {}).get('chunk_size', 500)
    context_size = args.context_size or CONFIG.get('story', {}).get('context_size', 200)
    
    # Generate the story
    story_text = generate_story(
        topic=topic,
        target_duration_minutes=target_duration,
        model_name=model_name,
        max_tokens_per_chunk=max_tokens,
        temperature=temperature,
        seed=seed,
        chunk_size=chunk_size,
        context_size=context_size
    )
    
    if story_text is None:
        log("Story generation failed", "❌")
        return 1
    
    # Save the story
    story_path = save_story(story_text, topic)
    if story_path is None:
        log("Failed to save story", "❌")
        return 1
    
    log("Story generation completed successfully!", "✅")
    return 0

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nProcess interrupted by user", "⚠️")
        log("Exiting gracefully...", "🛑")
        sys.exit(130)  # Standard exit code for SIGINT
