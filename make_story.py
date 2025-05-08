import lmstudio as lms
from pydantic import BaseModel
import random
import argparse
import yaml
import os
import sys
import time
from pathlib import Path

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
        # Use seed only if explicitly provided
        if seed is not None:
            print(f"🌱 Using seed: {seed}")
            seed_value = seed
        else:
            seed_value = None
        
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

def generate_topics(model):
    """Generate a list of diverse but calming story topics using the LLM.
    
    Args:
        model: The initialized LLM model
    
    Returns:
        List of generated topics
    """
    response = model.respond(
        """
        You are a creative assistant helping to generate bedtime story topics.

        Task:
        - Provide exactly 5 calm, slow-paced, sleep-friendly topics suitable for long, monotonous stories.
        - Include a gentle variety across settings: natural scenes, everyday routines, historical events, old objects, or imaginary calm places.
        - Avoid topics that are dramatic, violent, thrilling, or emotionally intense.
        - Each topic must include:
            • A short, poetic or slightly abstract title (2–6 words).
            • A one-sentence summary describing the type of story that could be told.
        - Example topics:
            • "The Journey of a River": Following the path of a river from its mountain source to the sea.
            • "Old Books on a Shelf": Exploring the quiet lives and histories of forgotten books.
            • "Moonlight Across Fields": A slow night walk under the moon in the countryside.
            • "The Village Bell Tower": Telling the quiet history of an old bell and the village it watches over.
            • "A Cat’s Afternoon Nap": Following the slow, dreamy thoughts of a cat drifting in and out of sleep.

        Format response strictly as JSON:
        {
          "topics": [
            { "title": "topic1", "summary": "summary1" },
            { "title": "topic2", "summary": "summary2" },
            { "title": "topic3", "summary": "summary3" },
            { "title": "topic4", "summary": "summary4" },
            { "title": "topic5", "summary": "summary5" }
          ]
        }
        """,
        response_format=Topics
    )
    
    topics = [Topic(**topic_dict) for topic_dict in response.parsed["topics"]]
    return topics

def generate_story(model, topic, target_duration=None):
    """Generate a long story by iteratively querying the LLM.
    
    Args:
        model: The LLM model to use
        topic: The topic object with title and summary
        target_duration: Target duration in minutes
        
    Returns:
        Dictionary containing the generated story data
    """
    if target_duration is None:
        target_duration = CONFIG.get('story', {}).get('target_duration_minutes', 60)

    chunks = []
    sentences_per_chunk = 10
    total_duration = 0
    all_sentences = []
    cumulative_summary = ""

    print(f"🛌 Generating story on topic: '{topic.title}'")
    print(f"📝 Topic summary: {topic.summary}")
    print(f"⏱️ Target duration: {target_duration} minutes")

    while total_duration < target_duration:
        # Calculate if we are starting, continuing, or finishing
        remaining_time = target_duration - total_duration
        phase = (
            "start" if total_duration == 0 else
            "finish" if remaining_time <= 5 else  # within last 5 minutes, wrap up
            "continue"
        )

        # Build context
        last_sentences = all_sentences[-5:] if len(all_sentences) >= 5 else all_sentences
        context_snippet = " ".join(last_sentences) or cumulative_summary or topic.summary

        # Build phase-specific instruction
        if phase == "start":
            phase_instruction = f"""
            Begin a gentle, sleep-inducing story about {topic.title}. Use this summary as a guide: {topic.summary}. Set the scene slowly, without jumping into events.
            Focus on calm atmosphere and subtle environmental details.
            """
        elif phase == "continue":
            phase_instruction = f"""
            Continue where the story left off, using the following context: "{context_snippet}".
            Progress the atmosphere and details gently, without rushing or repeating.
            """
        elif phase == "finish":
            phase_instruction = f"""
            Conclude the story softly, using the following context: "{context_snippet}".
            Provide a calm, satisfying wrap-up without abrupt endings or surprises.
            Let the reader drift off peacefully by the end.
            """

        prompt = f"""
        You are generating part of a continuous, calm, sleep-inducing story.

        Instructions:
        {phase_instruction}
        - Write {sentences_per_chunk} **new** slow-paced, non-exciting sentences.
        - Focus on gentle descriptions, minor environmental changes, and mild progression.
        - Avoid repeating earlier details.
        - Imagine the reader is drifting off to sleep, so avoid action or surprises.
        - Provide a brief summary (1–2 sentences) of only this **new** section.

        Response format (strict JSON):
        {{
            "sentences": ["sentence1", "sentence2", "..."],
            "short_summary": "summary here"
        }}
        """

        chunk = model.respond(prompt, response_format=StoryChunk)
        
        # Calculate duration based on configured sentence duration
        sentence_duration = CONFIG.get('story', {}).get('sentence_duration_minutes', 0.13)
        chunk_duration = len(chunk.parsed["sentences"]) * sentence_duration
        
        chunks.append(chunk.parsed)
        cumulative_summary += " " + chunk.parsed["short_summary"]
        total_duration += chunk_duration
        all_sentences.extend(chunk.parsed["sentences"])

        print(f"🧩 Preview: {chunk.parsed['sentences'][0]}")
        print(f"⏳ Current total duration: {total_duration}/{target_duration} minutes ({(total_duration/target_duration*100):.1f}%)")

    return {
        'topic_title': topic.title,
        'topic_summary': topic.summary,
        'total_duration': total_duration,
        'sentences': all_sentences
    }

def save_story_yaml(story_data):
    """Save the generated story to a YAML file.
    
    Args:
        story_data: Dictionary containing the story data
    
    Returns:
        Path to the saved story file
    """
    # Create a filename based on the topic title
    topic_title = story_data['topic_title']
    safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic_title)
    safe_topic = safe_topic.replace(" ", "_").lower()
    
    # Truncate if too long
    if len(safe_topic) > 50:
        safe_topic = safe_topic[:50]
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_topic}_{timestamp}.yaml"
    
    file_path = os.path.join(STORIES_DIR, filename)
    
    # Convert story data to a format suitable for YAML
    yaml_data = {
        'topic_title': story_data['topic_title'],
        'topic_summary': story_data['topic_summary'],
        'total_duration': story_data['total_duration'],
        'sentence_count': len(story_data['sentences']),
        'sentences': story_data['sentences']
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"💾 Story saved to: {file_path}")
    return file_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a long, monotonous story using an LLM")
    
    parser.add_argument("-t", "--topic", help="Story topic to use")
    parser.add_argument("-d", "--duration", type=int, help="Target duration in minutes")
    parser.add_argument("-m", "--model", help="Model to use (overrides config)")
    parser.add_argument("-s", "--seed", type=int, help="Random seed for reproducibility")
    
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
    seed = args.seed or CONFIG.get('llm', {}).get('seed')
    model = initialize_model(model_name, seed)
    
    if model is None:
        print("❌ Failed to initialize model")
        return 1
    
    # Get story parameters
    target_duration = args.duration or CONFIG.get('story', {}).get('target_duration_minutes', 60)
    
    try:
        # If topic is provided, use it; otherwise generate topics and select one
        if args.topic:
            # Create a Topic object from the provided string
            topic = Topic(title=args.topic, summary=f"A calming story about {args.topic}")
            print(f"🎯 Using provided topic: {topic.title}")
        else:
            # Generate topics and select one
            topics = generate_topics(model)
            print(f"🌙 Suggested topics:")
            for i, t in enumerate(topics):
                print(f"  {i+1}. {t.title} - {t.summary}")
            topic = random.choice(topics)
            print(f"🎯 Selected topic: {topic.title} - {topic.summary}")
        
        # Generate the story
        story_data = generate_story(model, topic, target_duration)
        
        # Save the story to a YAML file
        story_path = save_story_yaml(story_data)
        
        print(f"✅ Generated story with {len(story_data['sentences'])} sentences, total ~{story_data['total_duration']} minutes")
        
        # Clean up
        model.unload()
        return 0
    
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        print("🛑 Exiting gracefully...")
        model.unload()
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"❌ Error: {e}")
        model.unload()
        return 1

if __name__ == "__main__":
    sys.exit(main())
