# SleepTeller

A system to generate long, monotonous stories with voiceover and video files to help users fall asleep.

---

## 📖 Overview

**SleepTeller** is a Python-driven project designed to help people fall asleep by generating monotonous, low-stimulation stories on customizable topics.  
It combines:  
✅ iterative LLM-based story generation,  
✅ intelligent TTS narration with audio post-processing,  
✅ and seamless merging with looping background video.

---

## 🌟 Features

- **Iterative Story Generation**  
  Automatically handles context window limits by chaining LLM calls, continuing the story until the desired length is reached.

- **Thematic Control**  
  Supports configurable topics (e.g., countryside walks, spaceship maintenance, forgotten villages) to shape the storytelling mood.

- **Duration Targeting**  
  Lets you specify target duration (e.g., 20 min, 45 min) and keeps generating until that goal is met.

- **TTS Narration with Audio Processing**  
  Uses TTS to narrate the story in a calm, slow voice, with optional silence trimming and audio normalization.

- **Looping Visual Background**  
  Combines the final audio track with a looping background video (or static image) for a full audiovisual package.

- **Optional Background Music**  
  Allows adding subtle, non-distracting music for extra atmosphere.

- **Configurable Pipeline**  
  Each stage (generation → narration → video creation) can run independently or be chained together.

- **Organized Outputs**  
  Separates outputs into project folders for clean management.

---

## 🛠 Installation

1️⃣ Clone the repository:
```bash
git clone <repository-url>
cd sleepteller
2️⃣ Install Python dependencies:

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Set up an LLM API:
- Option 1: Run a local LM Studio server (default: http://localhost:1234/v1)
- Option 2: Use an external API (configure in your YAML file)

4️⃣ Set up a TTS server:
- Option 1: Use Zonos TTS (default: http://localhost:5001/generate)
- Option 2: Configure another TTS service in your YAML file

## 📁 Project Structure
```bash
sleepteller/
├── configs/               # YAML configs for different projects
├── lib/
│   ├── backgrounds/       # Background video or image loops
│   ├── music/             # Optional background music tracks
├── output/
│   └── {project_name}/    # Generated content per project
│       ├── stories/       # Full generated text files
│       ├── audio/         # TTS-generated audio files
│       └── videos/        # Final rendered videos
├── make_story.py          # Script for iterative story generation
├── make_audio.py          # Script for generating TTS audio
├── make_video.py          # Script for combining audio + visuals
├── sleepteller.py         # Main pipeline script
└── clean.py               # Cleanup utility
```
## ⚙️ Configuration
In your project YAML config (configs/sample.yaml by default), you can control:

### Story Settings
```yaml
story:
  topic: "a gentle walk through a quiet forest"
  target_duration_minutes: 20
  model: "gemma-3-12b-it-qat"
  max_tokens_per_chunk: 1000
  temperature: 0.7
```
### Voice Settings
```yaml
voice:
  tts_server: "http://localhost:5001/generate"
  voice_profile: "soft_female"
  speech_rate: 0.85
  normalization:
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
## 🧹 Cleanup
To clean generated outputs:

```bash
python clean.py --project your_project --all
```

Options:
- `--stories`: Remove generated story text files
- `--audio`: Remove generated audio files
- `--videos`: Remove final video files
- `--all`: Remove everything for the project

📌 License
MIT License

(c) 2025 SleepTeller Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so.

