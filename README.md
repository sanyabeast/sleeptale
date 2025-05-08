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

- **Dynamic Topic Generation**  
  Automatically generates calming, sleep-inducing topics with both titles and detailed summaries.

- **Duration Control**  
  Specify target duration in minutes (e.g., `-d 10`) and number of stories to generate (e.g., `-c 2`).

- **TTS Narration with Audio Processing**  
  Uses TTS to narrate the story with consistent -20dB audio normalization.

- **Random Visual Backgrounds**  
  Automatically selects random background videos from your library to create a visually calming experience.

- **Random Background Music**  
  Adds subtle, non-distracting music from your library with configurable volume.

- **Configurable Quality**  
  Adjustable video quality settings (e.g., `-q 0.25` for lower resolution) for different device requirements.

- **Simplified Configuration**  
  Uses a single `config.yaml` file in the root directory for all settings.

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
  tts_server: "http://localhost:5001/generate"
  voice_sample: "path/to/voice/sample.mp3"
  normalization:
    target_db: -20.0
    enabled: true
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
- `-c, --config`: Path to config file (default: configs/sample.yaml)
- `-t, --topic`: Override story topic
- `-d, --duration`: Override target duration in minutes
- `-m, --model`: Override LLM model
- `-s, --seed`: Set random seed for reproducibility

#### Audio Generation (make_audio.py)
- `-c, --config`: Path to config file (default: configs/sample.yaml)
- `-s, --story`: Path to specific story file (otherwise uses latest)
- `-f, --force`: Force regeneration even if audio exists

#### Video Generation (make_video.py)
- `-c, --config`: Path to config file (default: configs/sample.yaml)
- `-a, --audio`: Path to specific audio file (otherwise uses latest)
- `-b, --background`: Override background video/image
- `-m, --music`: Override background music file
- `--no-music`: Disable background music

## 📋 Example Workflow

### Complete Workflow

1. **Generate a story with a custom topic**:
   ```bash
   python make_story.py -t "floating on a calm ocean under a starry night sky" -d 30
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

1. **Generate voice lines for a specific story**:
   ```bash
   python make_voice_lines.py -s floating_on_a_calm_ocean_under_a_starry_night_sky_20250509_003012.yaml
   ```

2. **Create video for a specific story**:
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

