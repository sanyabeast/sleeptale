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
    
    # Create output directory structure
    output_dir = os.path.join(PROJECT_DIR, "output")
    STORIES_DIR = os.path.join(output_dir, "stories")
    
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
    recent_titles = get_recent_stories(10)
    recent_titles_str = '\n'.join([f'• "{title}"' for title in recent_titles])

    prompt = f"""
    You are a poetic assistant creating diverse, calming storytelling themes
    for a bedtime story generator.

    🎯 Your task:
    - Generate exactly 10 unique themes.
    - Each should inspire peaceful, slow, atmospheric stories.
    - Make them **visually intuitive** (scenes, objects, moods) and **emotionally neutral**.
    - Most themes should feel grounded or nature-based, but it's OK to include 1 space-themed or slightly surreal theme.

    💡 Source inspiration from:
    - Soft natural environments (meadows, rain, rivers, forests)
    - Quiet indoor scenes (attics, greenhouses, candlelit rooms)
    - Gentle weather (fog, snow, dusk light, breeze)
    - Simple routines (sorting, folding, watering, sweeping)
    - Passive observers (a cat, a lantern keeper, a gardener)
    - Rarely: cosmic isolation or slow drifting in space (e.g. “Orbital Window”)

    ❌ Avoid:
    - Abstract words (e.g., “resonance”, “refraction”, “entropy”)
    - Intellectual terms or metaphysical jargon
    - Anything intense, emotional, adventurous, or ominous

    🔒 Recently used stories (avoid too-similar ones):
    {recent_titles_str if recent_titles else '(No recent stories)'}

    📦 Respond strictly in this JSON format:
    {{
      "themes": ["theme1", "theme2", "theme3", "theme4", "theme5", "theme6", "theme7", "theme8", "theme9", "theme10"]
    }}
    """

    response = model.respond(prompt, response_format=ThemeList)
    return response.parsed["themes"]


def generate_topics_from_theme(model, theme):
    """Generate 10 grounded, calm story topics for a selected theme."""
    recent_titles = get_recent_stories(10)
    recent_titles_str = '\n'.join([f'• "{title}"' for title in recent_titles])

    prompt = f"""
    You are a calming creative assistant. Help generate gentle bedtime story topics
    under the theme: "{theme}".

    🎯 Task:
    - Suggest 10 story topics that feel peaceful, slow, and easy to visualize.
    - Focus on simple imagery: calm settings, slow natural rhythms, solitary routines.
    - Most topics should be grounded — include only 1–2 slightly surreal or cosmic ideas if appropriate to the theme.
    - Each topic must include a poetic **title** and a 1-sentence **summary**.

    💡 Good elements to use:
    - A quiet person or animal simply existing (e.g., feeding birds, watching clouds)
    - Nature moving slowly (e.g., snowfall, moss growing, fog shifting)
    - Small cozy spaces (e.g., attic, dock, greenhouse)
    - Soft tools or objects (e.g., a broom, notebook, candle)
    - Occasional dreamlike touches (e.g., drifting garden in orbit, forgotten telescope)

    ❌ Avoid:
    - Drama, dialogue, tension, or plot twists
    - Vague or overly abstract titles like “Temporal Fragments” or “Resonant Thresholds”
    - Anything hard to imagine or emotionally heavy

    🔒 Recently used topics:
    {recent_titles_str if recent_titles else '(No recent stories)'}

    📦 Strictly respond in JSON:
    {{
      "topics": [
        {{ "title": "topic1", "summary": "summary1" }},
        {{ "title": "topic2", "summary": "summary2" }},
        {{ "title": "topic3", "summary": "summary3" }},
        {{ "title": "topic4", "summary": "summary4" }},
        {{ "title": "topic5", "summary": "summary5" }},
        {{ "title": "topic6", "summary": "summary6" }},
        {{ "title": "topic7", "summary": "summary7" }},
        {{ "title": "topic8", "summary": "summary8" }},
        {{ "title": "topic9", "summary": "summary9" }},
        {{ "title": "topic10", "summary": "summary10" }}
      ]
    }}
    """

    response = model.respond(prompt, response_format=Topics)
    return [Topic(**topic_dict) for topic_dict in response.parsed["topics"]]


def generate_story(model, topic, theme=None, target_duration=None):
    """Generate a cohesive, slow-paced bedtime story from a given topic."""
    if target_duration is None:
        target_duration = CONFIG.get('story', {}).get('target_duration_minutes', 60)

    chunks, all_sentences = [], []
    total_duration, cumulative_summary = 0, ""
    sentences_per_chunk = 10

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
        'total_duration': 0,
        'sentences': []
    }, is_intermediate=True)

    while total_duration < target_duration:
        remaining = target_duration - total_duration
        phase = "start" if total_duration == 0 else "finish" if remaining <= 5 else "continue"
        recent_context = " ".join(all_sentences[-5:]) or topic.summary
        last_summary = chunks[-1]['short_summary'] if chunks else topic.summary

        prompt = f"""
        You are a poetic narrator writing a bedtime story in quiet, meditative style.

        🎯 Objective:
        - Calm the listener and lull them toward sleep.
        - Maintain soft pacing, ambient imagery, gentle rhythm.
        - Let the story feel timeless, without action or tension.

        ✅ You may include **a minimal character** (e.g., she, he, they, the keeper, the painter),
        but they must never speak or act dramatically.
        They may watch, fold, walk, feed, tidy, drift, wait, observe.
        Never use dialogue, conflict, or plot.

        🔁 Reinforce imagery from earlier chunks to improve cohesion.

        🧘‍♀️ Style:
        - Use tactile and sensory elements: fabric, paper, dust, shadows, candlelight, soft rain.
        - Let time stretch and blur — avoid strong transitions.
        - Focus on repetition, stillness, cyclical motions, fading light.

        Current phase: **{phase}**
        Theme: {theme or topic.title}
        Topic summary: {topic.summary}
        Context: "{recent_context}"
        So far: "{last_summary}"

        ✍️ Write exactly {sentences_per_chunk} new slow, descriptive sentences.
        📄 Then add a short summary (1–2 sentences) for just this section.

        Return in this exact JSON format:
        {{
          "sentences": ["..."],
          "short_summary": "..."
        }}
        """

        chunk = model.respond(prompt, response_format=StoryChunk)
        if not isinstance(chunk.parsed, dict) or "sentences" not in chunk.parsed:
            print("⚠️ Invalid chunk")
            continue

        sentences = [s for s in chunk.parsed["sentences"] if isinstance(s, str)]
        if not sentences:
            continue

        chunk_duration = len(sentences) * CONFIG.get('story', {}).get('sentence_duration_minutes', 0.13)
        chunks.append({"sentences": sentences, "short_summary": chunk.parsed.get("short_summary", "")})
        total_duration += chunk_duration
        all_sentences.extend(sentences)
        cumulative_summary += " " + chunk.parsed.get("short_summary", "")

        save_story_yaml({
            "topic_title": topic.title,
            "topic_summary": topic.summary,
            "theme": theme,
            "total_duration": total_duration,
            "sentences": all_sentences,
            "chunks": chunks,
            "cumulative_summary": cumulative_summary,
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
        "total_duration": total_duration,
        "sentence_count": len(all_sentences),
        "sentences": all_sentences
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
        'total_duration': story_data['total_duration'],
        'sentence_count': len(story_data['sentences']),
        'sentences': story_data['sentences']
    }

    # Add debug info for intermediate saves, remove it for final save
    if is_intermediate:
        if 'chunks' in story_data:
            yaml_data['debug'] = {
                'completion_percentage': story_data.get('completion_percentage', 0),
                'phase': story_data.get('phase', ''),
                'cumulative_summary': story_data.get('cumulative_summary', '')
            }
    else:
        # Remove debug info for final save
        yaml_data.pop('debug', None)

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
            story_data = generate_story(model, topic, theme, target_duration)
            
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
