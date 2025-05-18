import lmstudio as lms
from pydantic import BaseModel
import random
import argparse
import yaml
import os
import sys
import time
from pathlib import Path
import re
from datetime import datetime
from collections import Counter

# Get the absolute path of the project directory
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

# Global variables
CONFIG = None
STORIES_DIR = None

class StoryChunk(BaseModel):
    sentences: list[str]
    short_summary: str

class Topic(BaseModel):
    title: str
    summary: str

class Topics(BaseModel):
    topics: list[Topic]
    
class ThemeList(BaseModel):
    themes: list[str]

class Character(BaseModel):
    name: str
    role: str
    description: str = ""

class Chapters(BaseModel):
    exposition: str
    climax: str
    resolution: str

class StoryOutline(BaseModel):
    characters: list[Character]
    chapters: Chapters

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

def update_directories():
    """Update directory paths for output files."""
    global STORIES_DIR
    
    # Get directory paths from config
    output_dir_name = CONFIG.get('directories', {}).get('output', 'output')
    stories_dir_name = CONFIG.get('directories', {}).get('stories', 'stories')
    
    # Create output directory structure
    output_dir = os.path.join(PROJECT_DIR, output_dir_name)
    STORIES_DIR = os.path.join(output_dir, stories_dir_name)
    
    # Create directories if they don't exist
    os.makedirs(STORIES_DIR, exist_ok=True)
    
    print(f"📝 Stories directory: {STORIES_DIR}")

def initialize_model(model_name=None, seed=None):
    """Initialize the LLM model.
    
    Args:
        model_name: Name of the model to use
        seed: Random seed for reproducibility
    
    Returns:
        Initialized LLM model
    """
    if model_name is None:
        model_name = CONFIG.get('story', {}).get('model', 'gemma-3-4b-it-qat')
    
    print(f"🤖 Using model: {model_name}")
    
    try:
        # Use seed only if explicitly provided, otherwise use timestamp-based seed
        if seed is not None:
            print(f"🌱 Using provided seed: {seed}")
            seed_value = seed
        else:
            # Generate a seed based on current timestamp to ensure variety
            timestamp_seed = int(time.time()) % 100000
            seed_value = timestamp_seed
            print(f"🌱 Using timestamp-based seed: {seed_value}")
        
        # Initialize the model with the specified configuration
        model = lms.llm(model_name, config={
            "seed": seed_value,
            "temperature": CONFIG.get('story', {}).get('temperature', 0.7),
            "top_p": 0.9
        })
        return model
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        print("💡 Hint: Make sure LM Studio is running and the model is loaded.")
        return None

def get_recent_stories(num_stories=10):
    """Get titles of recently generated stories.
    
    Args:
        num_stories: Number of recent stories to retrieve
        
    Returns:
        List of story titles
    """
    story_files = sorted(
        Path(STORIES_DIR).glob('*.yaml'),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:num_stories]
    
    titles = []
    for story_file in story_files:
        try:
            with open(story_file, 'r', encoding='utf-8') as f:
                story_data = yaml.safe_load(f)
                if story_data and 'title' in story_data:
                    titles.append(story_data['title'])
        except Exception as e:
            print(f"⚠️ Warning: Could not read title from {story_file}: {e}")
    
    return titles

def generate_themes(model):
    """Generate 10 diverse, calming, and visually intuitive themes for sleep stories."""
    # Get recent stories to avoid repetition
    recent_titles = get_recent_stories(10)
    recent_titles_str = '\n'.join([f'• "{title}"' for title in recent_titles])
    
    # Get prompt from config and format with recent stories
    prompt = CONFIG.get('prompts', {}).get('theme', '')
    prompt = prompt.format(recent_stories=recent_titles_str if recent_titles else '(No recent stories)')
    
    # Get response from model
    response = model.respond(prompt, response_format=ThemeList)
    return response.parsed["themes"]


def generate_topics_from_theme(model, theme):
    """Generate 10 grounded, calm story topics for a selected theme."""
    # Get recent stories to avoid repetition
    recent_titles = get_recent_stories(10)
    recent_titles_str = '\n'.join([f'• "{title}"' for title in recent_titles])
    
    # Get prompt from config and format with theme and recent stories
    prompt = CONFIG.get('prompts', {}).get('topic', '')
    prompt = prompt.format(
        theme=theme,
        recent_stories=recent_titles_str if recent_titles else '(No recent stories)'
    )
    
    # Get response from model
    response = model.respond(prompt, response_format=Topics)
    return [Topic(**topic_dict) for topic_dict in response.parsed["topics"]]


def generate_story_outline(model, topic, theme=None):
    """Generate a structured outline with characters and three-chapter structure."""
    print(f"📋 Generating story outline for {topic.title}...")
    
    # Get prompt from config and format with theme and topic
    prompt = CONFIG.get('prompts', {}).get('outline', '')
    prompt = prompt.format(
        theme=theme or "",
        topic=topic.summary
    )
    
    # Get response from model
    response = model.respond(prompt, response_format=StoryOutline)
    outline = response.parsed
    
    # Extract character names safely - handle both object and dictionary access
    character_names = []
    for char in outline['characters']:
        if isinstance(char, dict):
            character_names.append(char.get('name', 'Unknown'))
        else:
            try:
                character_names.append(char.name)
            except AttributeError:
                character_names.append('Unknown')
    
    # Print outline summary
    print(f"👥 Characters: {', '.join(character_names)}")
    print(f"📖 Exposition: {outline['chapters']['exposition'][:50]}...")
    print(f"📈 Climax: {outline['chapters']['climax'][:50]}...")
    print(f"📉 Resolution: {outline['chapters']['resolution'][:50]}...")
    
    return outline

def compress_summary(model, text: str) -> str:
    """Compress a long summary into a concise representation to avoid repetition."""
    # Get prompt from config and format with text
    prompt = CONFIG.get('prompts', {}).get('compress_summary', '')
    prompt = prompt.format(text=text.strip())
    
    # Get the response as plain text without JSON parsing
    response = model.respond(prompt, response_format=None)
    # For LM Studio API, we can access the text directly
    response_text = str(response).strip()
    return response_text


# Removed repetition detection functions as they were deemed overengineering


def generate_story(model, topic, theme=None, target_duration=None, model_name=None):
    """Generate a cohesive, slow-paced bedtime story from a given topic using a three-chapter structure."""
    if target_duration is None:
        target_duration = CONFIG.get('story', {}).get('target_duration_minutes', 60)
        
    # Get model name for metadata
    if model_name is None:
        model_name = CONFIG.get('story', {}).get('model', 'unknown')

    # Generate story outline with characters and chapters
    outline = generate_story_outline(model, topic, theme)

    # Get chapter weights from config
    chapter_weights = CONFIG.get('story', {}).get('chapter_weights', {
        'exposition': 0.3, 
        'climax': 0.5, 
        'resolution': 0.2
    })
    
    # Calculate target duration for each chapter
    chapter_durations = {
        'exposition': target_duration * chapter_weights['exposition'],
        'climax': target_duration * chapter_weights['climax'],
        'resolution': target_duration * chapter_weights['resolution']
    }
    
    print(f"📊 Chapter durations (minutes):")
    print(f"📖 Exposition: {chapter_durations['exposition']:.1f}")
    print(f"📈 Climax: {chapter_durations['climax']:.1f}")
    print(f"📉 Resolution: {chapter_durations['resolution']:.1f}")

    chunks, all_sentences = [], []
    total_duration, cumulative_summary = 0, ""
    sentences_per_chunk = CONFIG.get('story', {}).get('sentences_per_chunk', 20)
    word_duration_seconds = CONFIG.get('story', {}).get('word_duration_seconds', 0.75)
    
    # Removed repetition detection logic as requested

    # Extract and format character information safely
    characters_info = []
    for char in outline['characters']:
        if isinstance(char, dict):
            name = char.get('name', 'Unknown')
            role = char.get('role', 'Unknown role')
            characters_info.append(f"{name} ({role})")
        else:
            try:
                characters_info.append(f"{char.name} ({char.role})")
            except AttributeError:
                characters_info.append('Unknown character')
    
    characters_info = ", ".join(characters_info)

    print(f"🛌 Generating story: {topic.title}")
    print(f"📝 Summary: {topic.summary}")
    if theme:
        print(f"🎨 Theme: {theme}")
    print(f"⏱️ Target: {target_duration} min")

    # Track the current chapter and chapter-specific duration
    current_chapter = "exposition"
    chapter_duration = 0

    # Create initial save path
    story_path = save_story_yaml({
        'topic_title': topic.title,
        'topic_summary': topic.summary,
        'theme': theme,
        'model': model_name,
        'outline': outline,
        'chapter_weights': chapter_weights,
        'total_duration': 0,
        'sentences': []
    }, is_intermediate=True)

    while total_duration < target_duration:
        # Determine story phase
        remaining = target_duration - total_duration
        phase = "start" if total_duration == 0 else "finish" if remaining <= 5 else "continue"
        
        # Check for chapter transitions
        if current_chapter == "exposition" and chapter_duration >= chapter_durations["exposition"]:
            current_chapter = "climax"
            chapter_duration = 0
            print(f"🔄 Transitioning to chapter: 📈 {current_chapter.upper()}")
        elif current_chapter == "climax" and chapter_duration >= chapter_durations["climax"]:
            current_chapter = "resolution"
            chapter_duration = 0
            print(f"🔄 Transitioning to chapter: 📉 {current_chapter.upper()}")
        
        # Compress the cumulative summary every few chunks to avoid repetition
        compressed_context = cumulative_summary
        if len(chunks) % 3 == 0 and cumulative_summary:
            compressed_context = compress_summary(model, cumulative_summary)
            # Also update the cumulative_summary with its compressed version to prevent endless growth
            cumulative_summary = compressed_context
        
        # Get recent context from the last few sentences
        recent_context = " ".join(all_sentences[-5:]) or topic.summary
        last_summary = chunks[-1]['short_summary'] if chunks else topic.summary
        
        # Repetition detection logic removed as requested
        avoid_elements = ""

        # Get tone modifiers from config
        tone_modifiers = CONFIG.get('story', {}).get('chapter_elements', {}).get('tone_modifiers', {
            "exposition": [
                "gentle morning light", "soft dawn mist", "awakening stillness",
                "first breath of day", "opening silence", "weightless beginning"
            ],
            "climax": [
                "golden dusk", "hidden resonance", "deeper awareness",
                "saturated moment", "present attention", "full-bodied presence"
            ],
            "resolution": [
                "settling dust", "quieting rhythm", "fading warmth",
                "dissolving light", "gentle release", "drifting toward silence"
            ]
        })
        
        # Use the tone modifiers for the current chapter, with fallback
        chapter_tone_modifiers = tone_modifiers.get(current_chapter, ["soft ambient glow"])
        tone_modifier = random.choice(chapter_tone_modifiers)
        
        # Get sensory focus elements from config
        sensory_focus = CONFIG.get('story', {}).get('chapter_elements', {}).get('sensory_focus', {
            "exposition": ["touch", "light", "scent", "texture", "air", "space"],
            "climax": ["sound", "warmth", "motion", "weight", "color", "presence"],
            "resolution": ["breath", "rhythm", "stillness", "shadow", "silence", "sleep"]
        })
        
        # Use the sensory focus for the current chapter, with fallback
        chapter_sensory_focus = sensory_focus.get(current_chapter, ["presence"])
        focus_sense = random.choice(chapter_sensory_focus)
        
        # Get current chapter description from outline
        chapter_description = outline['chapters'][current_chapter]
        
        # Get approaching_transition flag
        approaching_transition = False
        if current_chapter == "exposition" and chapter_duration >= chapter_durations["exposition"] * 0.8:
            approaching_transition = True
        elif current_chapter == "climax" and chapter_duration >= chapter_durations["climax"] * 0.8:
            approaching_transition = True
        
        # Get transition guidance if approaching a chapter boundary
        transition_guidance = ""
        if approaching_transition:
            next_chapter = "climax" if current_chapter == "exposition" else "resolution"
            next_chapter_desc = outline['chapters'][next_chapter]
            transition_guidance = f"\n\nYou are approaching the transition to the {next_chapter} chapter. Begin subtly shifting toward: {next_chapter_desc[:100]}..."
        
        # Get sentence length parameters from config
        sentence_length_target_range = CONFIG.get('story', {}).get('sentence_length_target_range', [8, 14])
        
        # Get prompt from config and format with all variables
        prompt = CONFIG.get('prompts', {}).get('story', '')
        prompt = prompt.format(
            avoid_elements=avoid_elements,
            sentences_per_chunk=sentences_per_chunk,
            phase=phase,
            theme=theme or topic.title,
            topic=topic.summary,
            story_essence=compressed_context,
            recent_context=recent_context,
            last_summary=last_summary,
            tone_modifier=tone_modifier,
            focus_sense=focus_sense,
            sentence_length_target_range=sentence_length_target_range,
            characters=characters_info,
            current_chapter=current_chapter,
            chapter_description=chapter_description,
            transition_guidance=transition_guidance
        )

        chunk = model.respond(prompt, response_format=StoryChunk)
        if not isinstance(chunk.parsed, dict) or "sentences" not in chunk.parsed:
            print("⚠️ Invalid chunk")
            continue

        sentences = [s for s in chunk.parsed["sentences"] if isinstance(s, str)]
        if not sentences:
            continue
            
        # Sentence deduplication logic removed as requested

        # Calculate duration based on word count
        word_count = sum(len(sentence.split()) for sentence in sentences)
        chunk_duration = (word_count * word_duration_seconds) / 60  # Convert to minutes
        chunks.append({
            "sentences": sentences, 
            "short_summary": chunk.parsed.get("short_summary", ""),
            "chapter": current_chapter
        })
        
        total_duration += chunk_duration
        chapter_duration += chunk_duration
        all_sentences.extend(sentences)
        cumulative_summary += " " + chunk.parsed.get("short_summary", "")

        save_story_yaml({
            "topic_title": topic.title,
            "topic_summary": topic.summary,
            "theme": theme,
            "model": model_name,
            "outline": outline,
            "total_duration": total_duration,
            "sentences": all_sentences,
            "chunks": chunks,
            "cumulative_summary": cumulative_summary,
            "compressed_context": compressed_context,
            "current_chapter": current_chapter,
            "chapter_duration": chapter_duration,
            "completion_percentage": total_duration / target_duration * 100,
            "phase": phase
        }, is_intermediate=True, existing_path=story_path)

        print(f"🧩 {sentences[0]}")
        print(f"⏳ {total_duration:.1f}/{target_duration} min | {current_chapter.capitalize()}: {chapter_duration:.1f}/{chapter_durations[current_chapter]:.1f} min")

    # Final write
    final = {
        "topic_title": topic.title,
        "topic_summary": topic.summary,
        "theme": theme,
        "model": model_name,
        "outline": outline,
        "total_duration": total_duration,
        "sentence_count": len(all_sentences),
        "sentences": all_sentences,
        "cumulative_summary": cumulative_summary,
        "compressed_context": compressed_context,
        "completion_percentage": 100,
        "phase": "complete"
    }

    save_story_yaml(final, is_intermediate=False, existing_path=story_path)
    return final


def save_story_yaml(story_data, is_intermediate=False, existing_path=None):
    """Save the generated story to a YAML file.
    
    Args:
        story_data: Dictionary containing the story data
        is_intermediate: If True, updates the existing file instead of creating a new one
        existing_path: Path to an existing file to update
    
    Returns:
        Path to the saved story file
    """
    # Create a filename based on the topic title
    topic_title = story_data['topic_title']
    safe_title = re.sub(r'[^a-zA-Z0-9]+', '_', topic_title.lower())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use existing path if provided, otherwise create new path
    if existing_path:
        file_path = existing_path
    else:
        file_path = os.path.join(STORIES_DIR, f"{safe_title}_{timestamp}.yaml")

    # Prepare data for YAML - keep the same structure as original
    yaml_data = {
        'topic_title': story_data['topic_title'],
        'topic_summary': story_data['topic_summary'],
        'model': story_data.get('model', CONFIG.get('story', {}).get('model', 'unknown')),  # Include model information
        'total_duration': story_data['total_duration'],
        'sentence_count': len(story_data['sentences']),
        'sentences': story_data['sentences']
    }
    
    # Include outline if it exists
    if 'outline' in story_data:
        yaml_data['outline'] = story_data['outline']
    
    # Include chapter weights if they exist
    if 'chapter_weights' in story_data:
        yaml_data['chapter_weights'] = story_data['chapter_weights']
    
    # Include current chapter information if it exists
    if 'current_chapter' in story_data:
        yaml_data['current_chapter'] = story_data['current_chapter']
        yaml_data['chapter_duration'] = story_data.get('chapter_duration', 0)

    # Add debug info for all saves
    if 'chunks' in story_data or ('cumulative_summary' in story_data and 'compressed_context' in story_data):
        yaml_data['debug'] = {
            'completion_percentage': story_data.get('completion_percentage', 0),
            'phase': story_data.get('phase', ''),
            'cumulative_summary': story_data.get('cumulative_summary', ''),
            'compressed_context': story_data.get('compressed_context', '')
        }

    # Save to YAML
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if not is_intermediate:
        print(f"💾 Story saved to: {file_path}")
    
    return file_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a long, monotonous story using an LLM")
    
    parser.add_argument("-t", "--topic", help="Story topic to use")
    parser.add_argument("-s", "--summary", help="Custom summary for the story (only used with --topic)")
    parser.add_argument("-d", "--duration", type=int, help="Target duration in minutes")
    parser.add_argument("-c", "--count", type=int, default=1, help="Number of stories to generate")
    parser.add_argument("-m", "--model", help="Model name or URL of the LM Studio API")
    parser.add_argument("--scale", type=float, default=1.0, 
                        help="Scale factor for max_tokens_per_chunk, chunk_size, context_size, and sentences_per_chunk")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--length", choices=['short', 'medium', 'long'], default='medium',
                        help="Story length preset (affects pacing and detail level)")
    parser.add_argument("--interactive", action="store_true", 
                        help="Enable interactive selection of themes and topics")
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    global CONFIG
    CONFIG = load_config()
    
    # Update directories
    update_directories()
    
    # Initialize the model
    model_name = args.model or CONFIG.get('story', {}).get('model')
    if not model_name:
        print("❌ No model specified. Use -m/--model or set 'model' in config.yaml")
        return 1

    seed = args.seed or CONFIG.get('llm', {}).get('seed')
    model = initialize_model(model_name, seed)
    
    if model is None:
        print("❌ Failed to initialize model")
        return 1
    
    # Get story parameters
    target_duration = args.duration or CONFIG.get('story', {}).get('target_duration_minutes', 60)
    
    # Apply scale factor to size-related parameters if provided
    if args.scale != 1.0:
        scale = args.scale
        print(f"🔍 Applying scale factor: {scale}")
        
        # Scale the parameters in the CONFIG
        if 'story' in CONFIG:
            if 'max_tokens_per_chunk' in CONFIG['story']:
                CONFIG['story']['max_tokens_per_chunk'] = int(CONFIG['story']['max_tokens_per_chunk'] * scale)
                print(f"  - max_tokens_per_chunk: {CONFIG['story']['max_tokens_per_chunk']}")
                
            if 'chunk_size' in CONFIG['story']:
                CONFIG['story']['chunk_size'] = int(CONFIG['story']['chunk_size'] * scale)
                print(f"  - chunk_size: {CONFIG['story']['chunk_size']}")
                
            if 'context_size' in CONFIG['story']:
                CONFIG['story']['context_size'] = int(CONFIG['story']['context_size'] * scale)
                print(f"  - context_size: {CONFIG['story']['context_size']}")
                
            if 'sentences_per_chunk' in CONFIG['story']:
                CONFIG['story']['sentences_per_chunk'] = int(CONFIG['story']['sentences_per_chunk'] * scale)
                print(f"  - sentences_per_chunk: {CONFIG['story']['sentences_per_chunk']}")
    
    try:
        # 3-stage generation pipeline
        for story_index in range(args.count):
            if args.count > 1:
                print(f"\
🔄 Generating story {story_index + 1} of {args.count}")
            
            # Stage 1: If topic is provided, use it; otherwise generate themes and topics
            if args.topic:
                # Create a Topic object from the provided string
                if args.summary:
                    summary = args.summary
                else:
                    summary = f"A calming story about {args.topic}"
                topic = Topic(title=args.topic, summary=summary)
                print(f"🎯 Using provided topic: {topic.title}")
                print(f"📝 Using summary: {topic.summary}")
                theme = None
            else:
                # Stage 1: Generate themes
                print("🌟 Generating themes...")
                themes = generate_themes(model)
                
                print(f"🎨 Available themes:")
                for i, theme in enumerate(themes):
                    print(f"  {i+1}. {theme}")
                
                # Select a theme (interactive or random)
                if args.interactive:
                    while True:
                        try:
                            theme_index = int(input("Select a theme (1-7): ")) - 1
                            if 0 <= theme_index < len(themes):
                                selected_theme = themes[theme_index]
                                break
                            else:
                                print(f"Please enter a number between 1 and {len(themes)}")
                        except ValueError:
                            print("Please enter a valid number")
                else:
                    selected_theme = random.choice(themes)
                
                print(f"🎨 Selected theme: {selected_theme}")
                
                # Stage 2: Generate topics based on the selected theme
                print("\
🧠 Generating topics for this theme...")
                topics = generate_topics_from_theme(model, selected_theme)
                
                print(f"🌙 Available topics:")
                for i, t in enumerate(topics):
                    print(f"  {i+1}. {t.title} - {t.summary}")
                
                # Select a topic (interactive or random)
                if args.interactive:
                    while True:
                        try:
                            topic_index = int(input("Select a topic (1-5): ")) - 1
                            if 0 <= topic_index < len(topics):
                                topic = topics[topic_index]
                                break
                            else:
                                print(f"Please enter a number between 1 and {len(topics)}")
                        except ValueError:
                            print("Please enter a valid number")
                else:
                    topic = random.choice(topics)
                
                print(f"🎯 Selected topic: {topic.title} - {topic.summary}")
                theme = selected_theme
            
            # Stage 3: Generate the story
            story_data = generate_story(model, topic, theme, target_duration, model_name=model_name)
            
            print(f"✅ Generated story with {len(story_data['sentences'])} sentences, total ~{story_data['total_duration']} minutes")
        
        # Clean up
        model.unload()
        return 0
    
    except KeyboardInterrupt:
        print("\
⚠️ Process interrupted by user")
        print("🛑 Exiting gracefully...")
        model.unload()
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"❌ Error: {e}")
        model.unload()
        return 1

if __name__ == "__main__":
    sys.exit(main())
