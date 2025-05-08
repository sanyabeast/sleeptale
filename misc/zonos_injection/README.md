# Zonos TTS Server Integration

## Overview

This module provides an HTTP server integration layer between DeepVideo2 and the Zonos Text-to-Speech (TTS) server. It enables DeepVideo2 to programmatically request voice generation with emotion control through a standardized REST API.

## Installation

The integration requires two simple steps:

1. **Copy the HTTP Server**: Copy `http_server.py` from this directory to your Zonos application directory.

2. **Modify the Zonos Application**: Update the Zonos `gradio_interface.py` file to initialize the HTTP server.

### Implementation Steps

#### 1. File Placement
```bash
cp http_server.py /path/to/your/zonos/app/
```

#### 2. Code Integration
Edit `gradio_interface.py` in your Zonos application and make the following changes:

**Add the import statement** at the top of the file with other imports:
```python
from http_server import InjectedServer
```

**Initialize the server** in the `build_interface()` function just before the `return demo` statement:
```python
injected_server = InjectedServer(custom_hook)
```

## Architecture

The integration follows a simple proxy pattern:

1. DeepVideo2 sends HTTP requests to the injected server on port 5001
2. The server translates these requests into appropriate function calls to the Zonos TTS engine
3. The Zonos engine generates the audio with the specified emotion
4. The server returns the generated audio data back to DeepVideo2

This architecture decouples DeepVideo2 from the Zonos implementation details, allowing for a clean API boundary.

## Usage

After installation:

1. Start the Zonos application (which will automatically initialize the HTTP server)
2. The server listens on `http://localhost:5001` by default
3. DeepVideo2 can send POST requests to `/generate` with the required parameters

The server accepts requests with the following parameters:
- `text`: The text to synthesize (required)
- `emotion`: The emotional tone to apply (optional)
- `rate`: Speech rate (optional)
- `samples`: Voice sample files (optional)

## Troubleshooting

Common issues to check:
- Verify `http_server.py` is in the correct directory
- Confirm both code modifications in `gradio_interface.py` are properly implemented
- Ensure the Zonos application is running before attempting to generate voice lines
- Check port 5001 is not being used by another application
