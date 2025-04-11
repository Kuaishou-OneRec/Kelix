# Kwai Video Visualization Tool

A web-based visualization tool for Kwai video data with responses.

## Features

- Upload and parse JSONL files containing video data and responses
- Display videos/images along with their corresponding responses
- Navigate through different items
- Modern and responsive UI

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `data` directory in the project root and place your key.json files there.

3. Run the Flask application:
```bash
python app.py
```

4. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Click the "Choose File" button to select a JSONL file
2. Click "Upload" to process the file
3. Use the "Previous" and "Next" buttons to navigate through the items
4. View the media (video/images) and responses for each item

## File Structure

- `app.py`: Flask application
- `templates/index.html`: Main HTML template
- `static/style.css`: CSS styles
- `static/script.js`: Frontend JavaScript
- `requirements.txt`: Python dependencies
- `data/`: Directory for key.json files
- `uploads/`: Directory for uploaded JSONL files

## Data Format

### JSONL Input Format
Each line should be a JSON object with the following structure:
```json
{
    "annotation": "",
    "responses": ["response1", "response2", ...],
    "source": "kwai_video",
    "__key__": "key_value"
}
```

### Key.json Format
Each key.json file should have the following structure:
```json
{
    "error": null,
    "media_path": "path/to/media",
    "media_type": "video",
    "pid": 123456789,
    "success": true,
    "text_fields": {
        "asr": "transcription",
        "caption": "caption text",
        "ocr": "ocr text",
        "text": "",
        "title": ""
    }
}
``` 