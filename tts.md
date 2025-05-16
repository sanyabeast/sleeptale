# Text-to-Speech (TTS) Implementation Guide

This document outlines the approach to working with different TTS providers and audio postprocessing techniques used in the SleepTale project. This architecture can be adapted for short video projects with similar voice generation requirements.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [TTS Provider System](#tts-provider-system)
   - [Base Provider Interface](#base-provider-interface)
   - [Provider Factory](#provider-factory)
   - [Supported Providers](#supported-providers)
3. [Audio Postprocessing Pipeline](#audio-postprocessing-pipeline)
   - [Tempo Manipulation](#tempo-manipulation)
   - [Echo Effect](#echo-effect)
   - [Audio Normalization](#audio-normalization)
4. [Implementation for Short Videos](#implementation-for-short-videos)
5. [Configuration](#configuration)

## Architecture Overview

The TTS system is designed with a modular architecture that separates concerns between:

1. **TTS Provider Interface**: Abstracts different TTS engines behind a common interface
2. **Audio Postprocessing Pipeline**: Applies effects to enhance the generated audio
3. **Configuration System**: Centralizes settings for both providers and effects

This separation allows for easy switching between TTS providers while maintaining consistent audio quality through standardized postprocessing.

## TTS Provider System

### Base Provider Interface

All TTS providers implement a common interface (`BaseTTSProvider`) with these key methods:

- `generate_voice_line(text, output_path, voice_options)`: Converts text to speech
- `preprocess_text(text)`: Sanitizes text for better TTS compatibility

This abstraction allows seamless switching between different TTS engines without changing the core application logic.

### Provider Factory

The system uses a factory pattern (`get_tts_provider()`) to instantiate the appropriate TTS provider based on configuration:

```python
def get_tts_provider(config):
    provider_name = config.get('provider', 'default_provider').lower()
    
    if provider_name == 'provider1':
        return Provider1(config)
    elif provider_name == 'provider2':
        return Provider2(config)
    # ...and so on
```

### Supported Providers

#### 1. Zonos Provider

- **Type**: Local API server
- **Features**:
  - Voice cloning from sample audio
  - Emotion control
  - Text preprocessing for better speech quality
- **Implementation**: Makes HTTP requests to a local Zonos TTS server

#### 2. Orpheus Provider

- **Type**: API-based
- **Features**:
  - Multiple voice presets
  - Speed control
  - Natural-sounding speech with good prosody
- **Implementation**: Communicates with Orpheus TTS server via REST API

#### 3. StyleTTS2 Provider

- **Type**: Gradio-based API
- **Features**:
  - High-quality voice synthesis
  - Voice preset selection
  - Speed adjustment
- **Implementation**: Uses Gradio client to interact with StyleTTS2 API

## Audio Postprocessing Pipeline

After generating raw voice audio, the system applies several postprocessing techniques to enhance quality and create the desired atmosphere:

### Tempo Manipulation

Two methods are implemented for tempo adjustment:

1. **Rubberband (Primary Method)**:
   - High-quality time-stretching without pitch distortion
   - Preserves voice naturalness even at slower speeds
   - Implemented via direct subprocess calls to the Rubberband CLI
   - Parameters:
     - `--tempo`: Controls playback speed (e.g., 0.9 = 90% speed)
     - `--pitch-hq`: High-quality pitch preservation
     - `--formant`: Preserves formants for more natural voice

2. **Librosa Fallback**:
   - Used when Rubberband is unavailable
   - Provides decent quality time-stretching
   - Implemented using `librosa.effects.time_stretch()`

### Echo Effect

A simple yet effective echo effect is implemented:

```python
def add_echo_effect(audio_data, sr, delay=0.3, decay=0.5):
    # Convert delay from seconds to samples
    delay_samples = int(delay * sr)
    
    # Create delayed version of the audio
    delayed_audio = np.zeros_like(audio_data)
    if delay_samples < len(audio_data):
        delayed_audio[delay_samples:] = audio_data[:-delay_samples] * decay
    
    # Mix original and delayed audio
    echo_audio = audio_data + delayed_audio
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(echo_audio))
    if max_val > 1.0:
        echo_audio = echo_audio / max_val * 0.99
        
    return echo_audio
```

Key parameters:
- `delay`: Time delay in seconds (typically 0.1-0.3s)
- `decay`: Volume reduction of the echo (0-1)

### Audio Normalization

Audio normalization ensures consistent volume levels across all voice lines:

```python
def normalize_audio(input_path, output_path=None, target_db=-20.0):
    # Load the audio file
    y, sr = librosa.load(input_path, sr=None)
    
    # Calculate current RMS energy
    rms = np.sqrt(np.mean(y**2))
    current_db = 20 * np.log10(rms)
    
    # Calculate the gain needed
    gain = 10 ** ((target_db - current_db) / 20)
    
    # Apply gain to normalize
    y_normalized = y * gain
    
    # Ensure we don't clip
    max_val = np.max(np.abs(y_normalized))
    if max_val > 1.0:
        y_normalized = y_normalized / max_val * 0.99
    
    # Export the normalized audio
    sf.write(output_path or input_path, y_normalized, sr)
```

## Implementation for Short Videos

For short video projects, this architecture can be adapted with these considerations:

1. **Voice Consistency**: 
   - Use a single voice preset/sample per video
   - Apply consistent postprocessing settings across all voice lines

2. **Segment-Based Processing**:
   - Divide script into logical segments (sentences or paragraphs)
   - Process each segment separately for easier editing and timing control

3. **Enhanced Workflow**:
   - Generate all voice lines first
   - Apply postprocessing effects
   - Integrate with video timeline

4. **Additional Effects for Videos**:
   - Consider adding compression for better audibility over background music
   - Implement EQ adjustments for voice clarity
   - Add subtle reverb for spatial context

## Configuration

The system uses a centralized YAML configuration:

```yaml
voice:
  # TTS provider to use
  provider: "provider_name"
  
  # Provider-specific settings
  provider_settings:
    # URL to TTS server
    tts_server: "http://localhost:5000/generate"
    # Voice samples/presets
    voice_presets: ["voice1", "voice2"]
    # Speed factor
    speed: 1.0
    # Sentences per voice line
    sentences_per_voice_line: 1
  
  # Audio post-processing settings
  postprocessing:
    # Audio normalization
    normalization:
      enabled: true
      target_db: -20.0
    # Tempo manipulation
    tempo:
      enabled: true
      factor: 0.9
    # Echo effect
    echo:
      enabled: true
      delay: 0.1
      decay: 0.1
    # Rubberband configuration
    rubberband:
      path: "/path/to/rubberband"
```

This configuration approach allows for easy adjustment of both TTS provider settings and audio postprocessing parameters without code changes.
