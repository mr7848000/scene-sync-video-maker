# AI Scene-Sync Video Maker

## Files
- `app.py`: Gradio web app
- `requirements.txt`: Python dependencies
- `render.yaml`: Render deployment configuration

## Deploy
1. Upload these three files to a GitHub repository.
2. In Render, create a new Web Service and connect the repository.
3. Render should detect `render.yaml`. If configuring manually:
   - Build command: `pip install -r requirements.txt`
   - Start command: `python app.py`
   - Free instance is only a best-effort option; AI transcription and 1080p encoding may exceed its RAM/time limits.
4. Open the deployed URL on your Android phone.

## How timing works
- Images are sorted by `Scene_<number>...jpg`.
- With AI timing enabled, Faster-Whisper's free local speech recognition generates word timestamps. The app distributes ordered images across the narration using approximate word-count chunks.
- This is speech-aware timing, not semantic image understanding. It cannot know whether a particular picture matches the meaning of a particular line unless scene narration/timestamps are supplied.
- If transcription fails or the host runs out of memory, disable AI timing to use equal-duration scenes.

## Supported inputs
- Audio: MP3, WAV, M4A, AAC
- Images: JPG/JPEG
- Output: 1920x1080 MP4, 16:9, 30 FPS

## Important
Render's free tier is not guaranteed to handle the model or longer renders. If it fails with memory errors, try equal-duration timing first or use a host with more RAM.
