# ContentRepurposer

Turn one piece of content into 5 platform-ready formats.

## Features

- **Audio Transcription**: Upload MP3/WAV/M4A files and auto-transcribe using AI
- **Text Input**: Paste any text content directly
- **Multi-Platform Output**: Generate content for Twitter/X, LinkedIn, Xiaohongshu, Blog, and Short Video Scripts
- **One-Click Copy**: Copy any generated content to clipboard instantly
- **Responsive Design**: Works on desktop and mobile

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py

# 3. Open in browser
# http://localhost:5000
```

## Tech Stack

- **Backend**: Python + Flask
- **AI Models**: SiliconFlow API (SenseVoiceSmall for transcription, DeepSeek-V3 for content generation)
- **Frontend**: Vanilla HTML/CSS/JS (no framework required)

## Project Structure

```
├── app.py              # Flask main application
├── templates/
│   └── index.html      # Frontend page
├── static/
│   └── style.css       # Styles
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## Configuration

API keys are configured directly in `app.py`. Update the following constants if needed:

- `API_KEY`: Your SiliconFlow API key
- `API_BASE_URL`: API base URL
- `TRANSCRIPTION_MODEL`: Audio transcription model
- `REWRITE_MODEL`: Content generation model
