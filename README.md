# SleepTeller

A system to generate long, monotonous stories with voiceover and video files to help users fall asleep.

**Author:** sanyabeast  
**License:** MIT

---

## 📖 Overview

**SleepTeller** is a Python-driven project designed to help people fall asleep by generating monotonous, low-stimulation stories on customizable topics.  
It combines:  
✅ iterative LLM-based story generation using LM Studio,  
✅ intelligent TTS narration with audio normalization,  
✅ and seamless merging with random background videos and music.

---

## 🌟 Features

- **Smart Story Generation**  
  Uses LLM to generate calming stories with consistent duration estimation (approximately 1.2 minutes per sentence).

- **Enhanced Topic Generation**  
  Intelligently generates diverse, calming topics across multiple categories (natural environments, everyday routines, dreamlike places, etc.) while avoiding repetition with recently used stories.

- **Duration Control**  
  Specify target duration in minutes (e.g., `-d 10`) and number of stories to generate (e.g., `-c 2`).

- **Multiple TTS Providers**  
  Supports multiple text-to-speech providers (Zonos, Orpheus, StyleTTS2) with provider-specific settings and easy switching via config or command line.  

- **Advanced Audio Processing**  
  Includes audio normalization, tempo adjustment, and echo effects for creating the perfect sleep-inducing narration.

- **Random Visual Backgrounds**  
  Automatically selects random background videos from your library to create a visually calming experience.

- **Random Background Music**  
  Adds subtle, non-distracting music from your library with configurable volume.

- **Configurable Quality**  
  Adjustable video quality settings (e.g., `-q 0.25` for lower resolution) for different device requirements.

- **Video Looping Utility**  
  Create seamlessly loopable videos from any source video using the `utils/make_loopable.py` script with customizable crossfade duration.

- **Simplified Configuration**  
  Uses a single `config.yaml` file in the root directory for all settings.

---

## 🛠️ Utility Scripts

### Make Loopable Videos

The `make_loopable.py` utility in the `utils` directory helps create seamlessly loopable videos by crossfading the end with the beginning:

```bash
python utils/make_loopable.py input_directory output_directory [--duration SECONDS]
```

**Parameters:**
- `input_directory`: Directory containing input videos
- `output_directory`: Directory to save processed videos
- `--duration`: Duration of the crossfade segment in seconds (default: 5.0)
- `--bitrate`: Override video bitrate (e.g., '4M')
- `--preset`: Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
- `--crf`: Constant Rate Factor (0-51, lower = better quality)

This utility is useful for creating background videos that can loop continuously without noticeable transitions.

---

## 📝 Notes

- **Story Generation**: The story generator creates monotonous, low-stimulation content by design. It avoids excitement, drama, or narrative tension.

- **Voice Selection**: For best results, use a calm, slightly monotone voice sample.

- **Background Videos**: Simple, non-distracting backgrounds work best. Avoid bright colors or rapid movement.

- **Background Music**: Select ambient, non-melodic tracks without vocals or strong rhythms.

- **Performance**: Story generation takes about 15-30 seconds per story.

- **Storage**: The final videos can be large. Use lower quality settings (e.g., `-q 0.25`) for smaller file sizes.

## 🔧 Requirements

- **Python 3.8+**
- **LM Studio** running locally with a loaded model
- **TTS Server** running locally
- **MoviePy** and its dependencies for video processing
- **librosa** and **soundfile** for audio processing

## 🚀 Quick Start

1. Make sure LM Studio and TTS Server are running
2. Create a `config.yaml` file in the root directory
3. Run `python make.py -d 10 -c 2 -q 0.25` to generate:
   - Two 10-minute stories
   - Voice lines for each story
   - Final videos at 25% quality

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**sanyabeast**
pip install -r requirements.txt
```

4️⃣ Set up the required services:
- **LM Studio**: Run a local LM Studio server (default: http://localhost:1234/v1)
- **TTS Server**: Set up a TTS server (default: http://localhost:5001/generate)

5️⃣ Prepare your media libraries:
```bash
# Create directories for background media
mkdir -p lib/videos lib/music lib/voice
```

6️⃣ Add your media files:
- Add background videos to `lib/videos/`
- Add background music to `lib/music/`
- Add voice samples to `lib/voice/`

## 📁 Project Structure
```bash
sleeptale/
├── config.yaml            # Main configuration file
├── lib/
│   ├── videos/            # Background videos
│   ├── music/             # Background music tracks
│   └── voice/             # Voice sample files
├── output/
│   ├── stories/           # Generated story YAML files
│   ├── voice_lines/       # Generated voice line audio files
│   └── videos/            # Final rendered videos
├── tts_providers/         # TTS provider implementations
│   ├── __init__.py        # Package initialization
│   ├── base_provider.py   # Base TTS provider interface
│   ├── provider_factory.py # Factory for creating providers
│   ├── zonos_provider.py  # Zonos TTS provider implementation
│   ├── orpheus_provider.py # Orpheus TTS provider implementation
│   └── utils.py           # Utility functions for providers
├── make_story.py          # Script for story generation
├── make_voice_lines.py    # Script for generating voice lines
└── make_video.py          # Script for creating final videos
```
## ⚙️ Configuration
The project uses a single configuration file (`config.yaml`) with the following sections:

### Story Generation Settings
```yaml
story:
  target_duration_minutes: 60
  model: "gemma-3-12b-it-qat"
  max_tokens_per_chunk: 1000
  temperature: 0.7
```

### Voice Generation Settings
```yaml
voice:
  # TTS provider to use (zonos, orpheus, styletts2)
  provider: "zonos"
  
  # Common settings for all providers
  sentences_per_voice_line: 1
  
  # Zonos provider settings
  zonos_settings:
    # URL to your Zonos TTS server
    tts_server: "http://localhost:5001/generate"
    # Voice samples to use
    voice_sample: 
      - "path/to/voice/sample1.mp3"
      - "path/to/voice/sample2.mp3"
    # Number of sentences to combine into a single voice line
    sentences_per_voice_line: 4
  
  # Orpheus provider settings
  orpheus_settings:
    # URL to your Orpheus TTS server
    tts_server: "http://localhost:5005/v1/audio/speech"
    # Voice presets to use
    voice_presets: ["jess", "mia", "leo", "zac"]
    # Speed factor (0.5 to 2.0)
    speed: 0.9
    # Number of sentences to combine into a single voice line
    sentences_per_voice_line: 4
    
  # StyleTTS2 provider settings
  styletts2_settings:
    # Base URL for the StyleTTS2 API
    base_url: "http://127.0.0.1:7860"
    # Voice presets to use (chosen randomly for each story)
    voice_presets: ["CalmDude1", "CalmGal1"]
    # Speed percentage (1-100)
    speed: 110
    # Number of sentences to combine into a single voice line
    sentences_per_voice_line: 1
  
  # Audio post-processing settings
  postprocessing:
    # Audio normalization settings
    normalization:
      enabled: true
      target_db: -20.0
    # Tempo manipulation settings
    tempo:
      enabled: false  # Set to true to enable
      factor: 0.9     # < 1.0 slows down, > 1.0 speeds up
    # Echo effect settings
    echo:
      enabled: true
      delay: 0.1      # Delay in seconds
      decay: 0.05     # Echo volume decay (0-1)
```

### Video Generation Settings
```yaml
video:
  start_delay: 2.0
  end_delay: 10.0
  line_delay: 1.0
  music_volume: 0.5
  resolution: "1920x1080"
```
    target_db: -20.0
    enabled: true
  silence_trimming:
    enabled: true
    max_silence_sec: 1.0
    threshold_db: -50
```
### Video Settings
```yaml
video:
  background_loop: "lib/backgrounds/soft_starry_night.mp4"
  add_music: true
  music_file: "lib/music/calm_ambient.mp3"
  music_volume: 0.2
  output_format: "mp4"
  resolution: "1920x1080"
```
## 🚀 Usage

### Complete Pipeline
Run the entire pipeline with a single command:
```bash
python sleepteller.py
```
By default, it will use `configs/sample.yaml` as the configuration file. You can specify a different config file with the `-c` option:
```bash
python sleepteller.py -c configs/your_custom_config.yaml
```

### Individual Steps
Or run each step separately:

1️⃣ Generate a Story
```bash
python make_story.py
```

2️⃣ Generate Audio Narration
```bash
python make_audio.py
```

3️⃣ Combine Into Final Video
```bash
python make_video.py
```

Each script will use `configs/sample.yaml` by default, or you can specify a different config file with the `-c` option.

### Command Line Options

#### Main Pipeline (sleepteller.py)
- `-c, --config`: Path to config file (default: configs/sample.yaml)
- `-t, --topic`: Override story topic
- `-d, --duration`: Override target duration in minutes
- `--skip-story`: Skip story generation step
- `--skip-audio`: Skip audio generation step
- `--skip-video`: Skip video generation step

#### Story Generation (make_story.py)
- `-t, --topic`: Override story topic
- `-s, --summary`: Custom summary for the story (only used with --topic)
- `-d, --duration`: Override target duration in minutes
- `-c, --count`: Number of stories to generate
- `-m, --model`: Override LLM model
- `--seed`: Set random seed for reproducibility
- `--length`: Story length preset (short, medium, long)

#### Voice Generation (make_voice_lines.py)
- `-s, --story`: Process only the specified story file
- `-v, --voice`: Force the use of a specific voice sample by index (0, 1, 2, etc.)
- `-p, --provider`: TTS provider to use (zonos, orpheus) - overrides config
- `--clean`: Remove all existing voice lines before generation
- `--force`: Force regeneration of voice lines even if they already exist
- `--normalize`: Force audio normalization even if disabled in config
- `--no-normalize`: Disable audio normalization even if enabled in config
- `--target-db`: Target dB level for audio normalization (overrides config)
- `--tempo`: Force tempo manipulation even if disabled in config
- `--no-tempo`: Disable tempo manipulation even if enabled in config
- `--tempo-factor`: Tempo change factor (< 1.0 slows down, > 1.0 speeds up)
- `--echo`: Force echo effect even if disabled in config
- `--no-echo`: Disable echo effect even if enabled in config
- `--echo-delay`: Echo delay in seconds
- `--echo-decay`: Echo decay factor (0-1)

#### Video Generation (make_video.py)
- `-c, --config`: Path to config file (default: configs/sample.yaml)
- `-a, --audio`: Path to specific audio file (otherwise uses latest)
- `-b, --background`: Override background video/image
- `-m, --music`: Override background music file
- `--no-music`: Disable background music

## 📋 Example Workflow

### Complete Workflow

1. **Generate a story with a custom topic and summary**:
   ```bash
   python make_story.py -t "floating on a calm ocean under a starry night sky" -s "A peaceful journey across gentle waves beneath an infinite cosmos" -d 30
   ```

2. **Generate voice lines for all stories**:
   ```bash
   python make_voice_lines.py
   ```

3. **Create videos with half resolution**:
   ```bash
   python make_video.py -q 0.5
   ```

### Processing a Specific Story

1. **Generate voice lines for a specific story with a specific voice**:
   ```bash
   python make_voice_lines.py -s floating_on_a_calm_ocean_under_a_starry_night_sky_20250509_003012.yaml -v 1
   ```

2. **Generate voice lines using a specific TTS provider**:
   ```bash
   python make_voice_lines.py -s floating_on_a_calm_ocean_under_a_starry_night_sky_20250509_003012.yaml -p orpheus
   ```

3. **Generate voice lines with audio effects**:
   ```bash
   python make_voice_lines.py -s floating_on_a_calm_ocean_under_a_starry_night_sky_20250509_003012.yaml --tempo --tempo-factor 0.85 --echo --echo-delay 0.2
   ```

4. **Create video for a specific story**:
   ```bash
   python make_video.py -s floating_on_a_calm_ocean_under_a_starry_night_sky_20250509_003012
   ```

### Regenerating Content

1. **Force regeneration of voice lines**:
   ```bash
   python make_voice_lines.py --force
   ```

2. **Force regeneration of videos**:
   ```bash
   python make_video.py -f
   ```

## 🧹 Cleanup
To clean generated outputs:

```bash
# On Windows
rd /s /q output
# On macOS/Linux
rm -rf output
```

## 🎥 Master Script

The `make.py` script automates the entire workflow from story generation to final video:

```bash
python make.py -c 3 -d 30 -q 0.75
```

Options:
- `-m, --model`: Model to use for story generation
- `-c, --count`: Number of stories to generate
- `-d, --duration`: Duration of stories in minutes
- `-q, --quality`: Quality of final video (1.0 = 1080p, 0.5 = 540p)

📌 License
MIT License

(c) 2025 sanyabeast

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so.

