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


def get_repetitive_ngrams(text: str, n=None, min_repeats=None):
    """Detect repeating phrases."""
    # Get parameters from config or use defaults
    if n is None:
        n = CONFIG.get('repetition_detection', {}).get('ngram', {}).get('size', 4)
    if min_repeats is None:
        min_repeats = CONFIG.get('repetition_detection', {}).get('ngram', {}).get('min_repeats', 2)
    
    words = text.lower().split()
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    counter = Counter(ngrams)
    return [ng for ng, c in counter.items() if c >= min_repeats]


def deduplicate_sentences(sentences, history, max_prefix=None):
    """Filter out sentences that start too similarly to recent history."""
    # Get parameters from config or use defaults
    if max_prefix is None:
        max_prefix = CONFIG.get('repetition_detection', {}).get('sentence_deduplication', {}).get('max_prefix', 5)
    
    def starts_like(s1, s2):
        # Compare the first few words to detect similar sentence beginnings
        return " ".join(s1.split()[:max_prefix]) == " ".join(s2.split()[:max_prefix])
    
    # Only keep sentences that don't start like any in the history
    return [s for s in sentences if all(not starts_like(s, h) for h in history)]


def generate_story(model, topic, theme=None, target_duration=None, model_name=None):
    """Generate a cohesive, slow-paced bedtime story from a given topic."""
    if target_duration is None:
        target_duration = CONFIG.get('story', {}).get('target_duration_minutes', 60)
        
    # Get model name for metadata
    if model_name is None:
        model_name = CONFIG.get('story', {}).get('model', 'unknown')

    chunks, all_sentences = [], []
    total_duration, cumulative_summary = 0, ""
    sentences_per_chunk = CONFIG.get('story', {}).get('sentences_per_chunk', 20)
    word_duration_seconds = CONFIG.get('story', {}).get('word_duration_seconds', 0.75)
    
    # Keep track of common repetitive phrases to avoid
    repetitive_elements = set()

    print(f"🛌 Generating story: {topic.title}")
    print(f"📝 Summary: {topic.summary}")
    if theme:
        print(f"🎨 Theme: {theme}")
    print(f"⏱️ Target: {target_duration} min")

    # Create initial save path
    story_path = save_story_yaml({
        'topic_title': topic.title,
        'topic_summary': topic.summary,
        'theme': theme,
        'model': model_name,  # Include model information
        'total_duration': 0,
        'sentences': []
    }, is_intermediate=True)

    while total_duration < target_duration:
        remaining = target_duration - total_duration
        phase = "start" if total_duration == 0 else "finish" if remaining <= 5 else "continue"
        
        # Compress the cumulative summary every few chunks to avoid repetition
        compressed_context = cumulative_summary
        if len(chunks) % 3 == 0 and cumulative_summary:
            compressed_context = compress_summary(model, cumulative_summary)
            # Also update the cumulative_summary with its compressed version to prevent endless growth
            cumulative_summary = compressed_context
        
        # Get recent context from the last few sentences
        recent_context = " ".join(all_sentences[-5:]) or topic.summary
        last_summary = chunks[-1]['short_summary'] if chunks else topic.summary
        
        # Identify repetitive elements to avoid
        avoid_elements = ""
        if len(all_sentences) > 10:
            recent_text = " ".join(all_sentences[-20:])
            
            # Detect repetitive phrases using n-grams
            repeated = get_repetitive_ngrams(recent_text)
            if repeated:
                avoid_elements += "\n⚠️ Avoid repeating these ideas: " + ", ".join(repeated)
            
            # Track overused words
            word_counts = Counter(recent_text.lower().split())
            min_length = CONFIG.get('repetition_detection', {}).get('word_frequency', {}).get('min_length', 3)
            min_frequency = CONFIG.get('repetition_detection', {}).get('word_frequency', {}).get('min_frequency', 3)
            max_words = CONFIG.get('repetition_detection', {}).get('word_frequency', {}).get('max_words', 5)
            
            frequent_words = [w for w, c in word_counts.items() if len(w) > min_length and c > min_frequency][:max_words]
            if frequent_words:
                avoid_elements += "\n🚫 Too frequent: " + ", ".join(frequent_words)
            
            # Simple detection of common phrases
            text = " ".join(all_sentences[-20:]).lower()
            common_phrases = CONFIG.get('repetition_detection', {}).get('common_phrases', [
                "dust motes", "motes of dust", "shaft of light", "beam of light",
                "sunlight", "moonlight", "candlelight", "shadows", "window"
            ])
            for phrase in common_phrases:
                if text.count(phrase) >= 2:
                    repetitive_elements.add(phrase)
        
        if repetitive_elements:
            avoid_elements += "\n⚠️ Avoid repeating these overused elements: " + ", ".join(repetitive_elements) + "."

        # Generate random tone modifier and focus sense for variety
        tone_modifier = random.choice([
            "golden dusk", "linen-soft twilight", "foggy glass morning",
            "moonlight across old floorboards", "drifting cabin glow",
            "warm attic haze", "still lake reflection"
        ])
        
        focus_sense = random.choice([
            "touch", "sound", "warmth", "motion", "air", "texture"
        ])
        
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
            sentence_length_target_range=sentence_length_target_range
        )


        chunk = model.respond(prompt, response_format=StoryChunk)
        if not isinstance(chunk.parsed, dict) or "sentences" not in chunk.parsed:
            print("⚠️ Invalid chunk")
            continue

        sentences = [s for s in chunk.parsed["sentences"] if isinstance(s, str)]
        if not sentences:
            continue
            
        # Filter out sentences that are too similar to recent ones
        if len(all_sentences) > 10:
            max_prefix = CONFIG.get('repetition_detection', {}).get('sentence_deduplication', {}).get('max_prefix', 5)
            min_keep_ratio = CONFIG.get('repetition_detection', {}).get('sentence_deduplication', {}).get('min_keep_ratio', 0.7)
            
            filtered_sentences = deduplicate_sentences(sentences, all_sentences[-20:], max_prefix=max_prefix)
            # Only use filtered sentences if we didn't lose too many
            if len(filtered_sentences) >= len(sentences) * min_keep_ratio:
                sentences = filtered_sentences

        # Calculate duration based on word count
        word_count = sum(len(sentence.split()) for sentence in sentences)
        chunk_duration = (word_count * word_duration_seconds) / 60  # Convert to minutes
        chunks.append({"sentences": sentences, "short_summary": chunk.parsed.get("short_summary", "")})
        total_duration += chunk_duration
        all_sentences.extend(sentences)
        cumulative_summary += " " + chunk.parsed.get("short_summary", "")

        save_story_yaml({
            "topic_title": topic.title,
            "topic_summary": topic.summary,
            "theme": theme,
            "model": model_name,  # Include model information
            "total_duration": total_duration,
            "sentences": all_sentences,
            "chunks": chunks,
            "cumulative_summary": cumulative_summary,
            "compressed_context": compressed_context,
            "completion_percentage": total_duration / target_duration * 100,
            "phase": phase
        }, is_intermediate=True, existing_path=story_path)

        print(f"🧩 {sentences[0]}")
        print(f"⏳ {total_duration:.1f}/{target_duration} min")

    # Final write
    final = {
        "topic_title": topic.title,
        "topic_summary": topic.summary,
        "theme": theme,
        "model": model_name,  # Include model information
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
