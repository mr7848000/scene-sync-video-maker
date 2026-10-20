import os
import re
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import imageio_ffmpeg
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

APP_TITLE = "AI Scene-Sync Video Maker"
PORT = int(os.environ.get("PORT", "7860"))
VIDEO_W, VIDEO_H, FPS = 1920, 1080, 30


def run_cmd(args, timeout=None):
    result = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "Unknown command error")[-5000:]
        raise RuntimeError(message)
    return result


def get_audio_duration(audio_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # ffmpeg prints the input metadata, including Duration, to stderr.
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(audio_path), "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError("Audio duration read nahi ho paayi. MP3 ya WAV file try karein.")
    hours, minutes, seconds = int(match.group(1)), int(match.group(2)), float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def scene_number(path):
    name = Path(path).name
    match = re.search(r"scene[\s_-]*(\d+)", name, re.IGNORECASE)
    if match:
        return (0, int(match.group(1)), name.lower())
    # Put files without Scene_N in natural-ish filename order after numbered files.
    nums = re.findall(r"\d+", name)
    return (1, int(nums[0]) if nums else 10**12, name.lower())


def sort_images(files):
    paths = [str(x) for x in (files or [])]
    paths.sort(key=scene_number)
    return paths


def transcribe_and_get_word_times(audio_path, model_size="tiny"):
    """
    Optional free local transcription. The tiny multilingual model is used to
    keep resource use lower, but it can still exceed a free web host's memory.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(audio_path), beam_size=1, vad_filter=True, word_timestamps=True
    )
    word_times = []
    transcript_parts = []
    for seg in segments:
        transcript_parts.append(seg.text.strip())
        words = getattr(seg, "words", None) or []
        if words:
            for word in words:
                token = (word.word or "").strip()
                if token:
                    start = max(0.0, float(word.start))
                    end = max(start + 0.02, float(word.end))
                    word_times.append((start, end, token))
        else:
            # Fallback for segments where word-level timestamps are unavailable.
            tokens = seg.text.strip().split()
            if tokens:
                span = max(0.05, float(seg.end) - float(seg.start))
                for i, token in enumerate(tokens):
                    start = float(seg.start) + span * i / len(tokens)
                    end = float(seg.start) + span * (i + 1) / len(tokens)
                    word_times.append((start, end, token))
    return word_times, " ".join(transcript_parts), getattr(info, "language", "unknown")


def boundaries_from_words(word_times, image_count, total_duration):
    """
    Split the ordered narration into approximately equal word-count chunks.
    This makes scene duration speech-aware, but it cannot semantically know
    what an image depicts. Images are matched in filename order.
    """
    if not word_times or image_count < 2:
        return [i * total_duration / image_count for i in range(image_count + 1)]

    # Sort and clamp timestamps to the audio duration.
    clean = []
    for start, end, token in word_times:
        start = min(total_duration, max(0.0, float(start)))
        end = min(total_duration, max(start + 0.02, float(end)))
        clean.append((start, end, token))
    clean.sort(key=lambda x: x[0])

    if len(clean) < image_count:
        return [i * total_duration / image_count for i in range(image_count + 1)]

    # Find a timestamp at each target word quantile, then keep boundaries monotonic.
    bounds = [0.0]
    n_words = len(clean)
    for i in range(1, image_count):
        target_index = min(n_words - 1, round(i * n_words / image_count))
        # Use the midpoint of the gap around the target word to avoid cutting through speech.
        target_start = clean[target_index][0]
        previous_end = clean[target_index - 1][1] if target_index > 0 else 0.0
        boundary = (previous_end + target_start) / 2.0
        bounds.append(boundary)
    bounds.append(total_duration)

    # Ensure useful positive durations; if timestamps collapse, use equal spacing.
    min_duration = 0.25
    if any(bounds[i + 1] - bounds[i] < min_duration for i in range(len(bounds) - 1)):
        bounds = [i * total_duration / image_count for i in range(image_count + 1)]
    return bounds


def normalize_image(src, dest):
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        fitted = ImageOps.contain(im, (VIDEO_W, VIDEO_H), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (VIDEO_W, VIDEO_H), (0, 0, 0))
        x = (VIDEO_W - fitted.width) // 2
        y = (VIDEO_H - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        canvas.save(dest, "JPEG", quality=92, optimize=True)


def make_video(audio_file, image_files, ai_sync, output_quality):
    if not audio_file:
        raise gr.Error("Pehle audio file upload karein.")
    images = sort_images(image_files)
    if not images:
        raise gr.Error("Kam se kam ek JPG/JPEG image upload karein.")

    valid_images = []
    for path in images:
        if Path(path).suffix.lower() in (".jpg", ".jpeg"):
            valid_images.append(path)
    images = valid_images
    if not images:
        raise gr.Error("Sirf JPG/JPEG images supported hain.")

    work_dir = Path(tempfile.mkdtemp(prefix="scene_sync_"))
    try:
        audio_path = Path(audio_file)
        duration = get_audio_duration(audio_path)
        if duration <= 0:
            raise RuntimeError("Audio duration invalid hai.")

        notes = []
        if ai_sync:
            try:
                word_times, transcript, language = transcribe_and_get_word_times(audio_path)
                if word_times:
                    bounds = boundaries_from_words(word_times, len(images), duration)
                    notes.append(
                        f"Speech-aware timing used. Detected language: {language}. "
                        f"Transcribed words: {len(word_times)}."
                    )
                else:
                    bounds = [i * duration / len(images) for i in range(len(images) + 1)]
                    notes.append("Speech timestamps nahi mile; equal timing use ki gayi.")
            except Exception as e:
                bounds = [i * duration / len(images) for i in range(len(images) + 1)]
                notes.append(
                    "AI transcription run nahi ho paayi; equal timing fallback use hua. "
                    f"Reason: {str(e)[:350]}"
                )
        else:
            bounds = [i * duration / len(images) for i in range(len(images) + 1)]
            notes.append("Equal-duration timing selected.")

        normalized_dir = work_dir / "images"
        normalized_dir.mkdir()
        normalized = []
        for idx, source in enumerate(images, start=1):
            dest = normalized_dir / f"image_{idx:04d}.jpg"
            normalize_image(source, dest)
            normalized.append(dest)

        # FFmpeg concat demuxer: each normalized still image is displayed for its scene duration.
        concat_path = work_dir / "images.txt"
        with concat_path.open("w", encoding="utf-8") as f:
            for idx, image_path in enumerate(normalized):
                scene_duration = max(0.05, bounds[idx + 1] - bounds[idx])
                escaped = str(image_path).replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
                f.write(f"duration {scene_duration:.4f}\n")
            # Repeat final image so concat demuxer honors the final duration.
            escaped = str(normalized[-1]).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

        output_path = work_dir / "scene_sync_video.mp4"
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        preset = "ultrafast" if output_quality == "Faster / smaller CPU load" else "veryfast"
        crf = "27" if output_quality == "Faster / smaller CPU load" else "23"

        run_cmd([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-i", str(audio_path),
            "-vf", f"fps={FPS},format=yuv420p",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            "-shortest", str(output_path)
        ], timeout=3600)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        status = (
            f"Video ready! Images: {len(images)} | Audio: {duration/60:.1f} min | "
            f"Output: 1920x1080, 16:9, 30 FPS | Size: {size_mb:.1f} MB\n"
            + "\n".join(notes)
            + "\nNote: Images are matched in Scene-number order. Speech-aware timing distributes "
              "images across the narration by word count; it does not understand image meaning. "
              "For exact semantic matching, each scene needs its narration text or manually supplied timestamps."
        )
        return str(output_path), status
    except gr.Error:
        raise
    except Exception as e:
        details = traceback.format_exc()[-2500:]
        raise gr.Error(f"Video create nahi ho paayi: {str(e)}\n\nTechnical details:\n{details}")


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(
        "# 🎬 AI Scene-Sync Video Maker\n"
        "Audio aur Scene-number wali JPG/JPEG images upload karein. Tool images ko filename ke "
        "scene number ke hisaab se sort karke MP4 banayega.\n\n"
        "**Important:** AI timing speech ke word timestamps se image durations distribute karti hai. "
        "Ye image ke visual meaning ko automatically samajhkar dialogue se match nahi karti."
    )
    with gr.Row():
        audio_input = gr.File(
            label="Audio file (MP3/WAV)", file_types=[".mp3", ".wav", ".m4a", ".aac"], type="filepath"
        )
        images_input = gr.File(
            label="Scene images (JPG/JPEG — multiple files)", file_count="multiple",
            file_types=[".jpg", ".jpeg"], type="filepath"
        )
    ai_sync_input = gr.Checkbox(
        value=True,
        label="Use free AI speech recognition for speech-aware scene timing (slower; may need more RAM)"
    )
    quality_input = gr.Radio(
        choices=["Faster / smaller CPU load", "Better quality"],
        value="Faster / smaller CPU load",
        label="Rendering setting"
    )
    create_button = gr.Button("Create MP4 Video", variant="primary")
    video_output = gr.File(label="Download your MP4 video")
    status_output = gr.Textbox(label="Status", lines=7)
    create_button.click(
        fn=make_video,
        inputs=[audio_input, images_input, ai_sync_input, quality_input],
        outputs=[video_output, status_output]
    )
    gr.Markdown(
        "### Naming example\n"
        "`Scene_1_20261019143419.jpg`, `Scene_2_20261019143419.jpg`, etc.\n\n"
        "### Hosting note\n"
        "Free Render instances can run out of RAM or time out during AI transcription or 1080p rendering. "
        "If that happens, uncheck AI speech recognition and use equal-duration timing, or use a host with more RAM."
    )

if __name__ == "__main__":
    demo.queue(max_size=2).launch(server_name="0.0.0.0", server_port=PORT)
