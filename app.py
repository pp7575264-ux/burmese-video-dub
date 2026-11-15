import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Dict

import edge_tts
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI(title="Burmese AI Video Dubbing (Cloud Edition)")

BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "workspace"
STATIC_DIR = BASE_DIR / "static"
WORK_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
SERVER_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

jobs: Dict[str, Dict[str, Any]] = {}


class SegmentItem(BaseModel):
    start_sec: float
    end_sec: float
    orig_text: str
    my_text: str


class DubScript(BaseModel):
    detected_language: str
    segments: list[SegmentItem]


def run_cmd(cmd: list[str]) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command error: {res.stderr[-500:]}")
    return res.stdout.strip()


def get_media_duration(file_path: str) -> float:
    out = run_cmd([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path,
    ])
    return float(out)


def get_video_codec(file_path: str) -> str:
    return run_cmd([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path,
    ]).strip().lower()


def has_audio_stream(file_path: str) -> bool:
    out = run_cmd([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path,
    ])
    return "audio" in out.lower()


def build_atempo_filter(speed: float) -> str:
    speed = max(0.5, min(speed, 1.85))
    filters = []
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    filters.append(f"atempo={speed:.4f}")
    return ",".join(filters)


def cleanup_old_jobs(max_age_sec: int = 3600) -> None:
    now = time.time()
    for p in WORK_DIR.iterdir():
        if p.is_dir() and (now - p.stat().st_mtime) > max_age_sec:
            shutil.rmtree(p, ignore_errors=True)


async def generate_tts_segments(segments: list[dict], voice: str, seg_dir: Path) -> None:
    for idx, seg in enumerate(segments):
        text = seg["my_text"].strip()
        if not text:
            seg["tts_mp3"] = None
            continue
        mp3_path = seg_dir / f"seg_{idx:04d}.mp3"
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(mp3_path))
        seg["tts_mp3"] = str(mp3_path)


def read_wav_mono_int16(wav_path: str) -> np.ndarray:
    with wave.open(wav_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        return np.frombuffer(frames, dtype=np.int16)


def write_wav_mono_int16(wav_path: str, data: np.ndarray, sample_rate: int = 24000) -> None:
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def gemini_transcribe_and_translate(audio_mp3: Path, source_lang: str, api_key: str) -> dict:
    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=str(audio_mp3))
    try:
        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        lang_hint = (
            "Auto-detect the spoken language."
            if source_lang == "auto"
            else f"The spoken language is '{source_lang}'."
        )

        prompt = (
            f"Listen carefully to this audio track. {lang_hint} "
            "Break all spoken dialogue into natural subtitle/voiceover segments with accurate "
            "start_sec and end_sec timestamps (in seconds, as floats). "
            "For each segment, provide: "
            "1) orig_text: exact transcription in the original language, "
            "2) my_text: natural, conversational Burmese (Myanmar Unicode) translation suitable for voiceover dubbing "
            "(keep Burmese sentences concise enough to fit roughly within the segment duration). "
            "Skip silent music-only parts."
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, uploaded],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DubScript,
                temperature=0.2,
            ),
        )
        return json.loads(response.text)
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass


def process_video_pipeline(
    job_id: str, input_video_path: Path, source_lang: str, voice: str, api_key: str
) -> None:
    job_dir = input_video_path.parent
    seg_dir = job_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    try:
        cleanup_old_jobs()
        jobs[job_id].update({"progress": 10, "status": "Video ဖိုင်ကို စစ်ဆေးနေပါသည်..."})

        if not has_audio_stream(str(input_video_path)):
            raise RuntimeError("Uploaded video တွင် မူရင်းအသံ (Audio track) မပါဝင်ပါ။")

        video_duration = get_media_duration(str(input_video_path))
        extracted_mp3 = job_dir / "extracted_audio.mp3"

        run_cmd([
            "ffmpeg", "-y", "-i", str(input_video_path),
            "-vn", "-acodec", "libmp3lame", "-b:a", "64k", "-ar", "16000", "-ac", "1",
            str(extracted_mp3),
        ])

        jobs[job_id].update({
            "progress": 30,
            "status": "Gemini AI ဖြင့် စကားသံများကို မိနစ်အတိအကျ ခွဲထုတ်ပြီး မြန်မာဘာသာပြန်နေပါသည်...",
        })
        script_data = gemini_transcribe_and_translate(extracted_mp3, source_lang, api_key)

        segments = [
            {
                "start": max(0.0, float(s["start_sec"])),
                "end": max(0.1, float(s["end_sec"])),
                "orig_text": s["orig_text"].strip(),
                "my_text": s["my_text"].strip(),
            }
            for s in script_data.get("segments", [])
            if s.get("my_text", "").strip()
        ]
        segments.sort(key=lambda x: x["start"])

        if not segments:
            raise RuntimeError("Video ထဲတွင် စကားပြောသံ မတွေ့ရှိပါ။")

        jobs[job_id].update({
            "progress": 55,
            "status": f"မြန်မာ AI Voiceover အသံသွင်းနေပါသည် ({len(segments)} ကြောင်း)...",
            "detected_lang": script_data.get("detected_language", ""),
            "segments": segments,
        })
        asyncio.run(generate_tts_segments(segments, voice, seg_dir))

        jobs[job_id].update({
            "progress": 78,
            "status": "Voiceover အသံများကို မူရင်း Video မိနစ်/စက္ကန့်များနှင့် ကိုက်ညီအောင် Sync လုပ်နေပါသည်...",
        })
        # 24kHz mono matches Edge-TTS native quality and keeps RAM tiny for Render 512MB tier
        sample_rate = 24000
        total_samples = int(video_duration * sample_rate) + sample_rate
        master_audio = np.zeros(total_samples, dtype=np.int32)

        for idx, seg in enumerate(segments):
            mp3_path = seg.get("tts_mp3")
            if not mp3_path or not os.path.exists(mp3_path):
                continue

            tts_dur = get_media_duration(mp3_path)
            if tts_dur <= 0.05:
                continue

            next_start = segments[idx + 1]["start"] if idx + 1 < len(segments) else video_duration
            available_dur = max(0.3, (next_start - seg["start"]) - 0.04)
            speed = (tts_dur / available_dur) if tts_dur > available_dur else 1.0

            synced_wav = seg_dir / f"synced_{idx:04d}.wav"
            cmd = ["ffmpeg", "-y", "-i", mp3_path]
            if speed > 1.02:
                cmd += ["-filter:a", build_atempo_filter(speed)]
            cmd += ["-ar", str(sample_rate), "-ac", "1", "-acodec", "pcm_s16le", str(synced_wav)]
            run_cmd(cmd)

            seg_audio = read_wav_mono_int16(str(synced_wav)).astype(np.int32)
            start_sample = int(seg["start"] * sample_rate)
            end_sample = min(start_sample + len(seg_audio), total_samples)
            insert_len = end_sample - start_sample
            if insert_len > 0:
                master_audio[start_sample:end_sample] += seg_audio[:insert_len]

        dubbed_wav = job_dir / "dubbed_master.wav"
        write_wav_mono_int16(
            str(dubbed_wav), np.clip(master_audio, -32768, 32767).astype(np.int16), sample_rate
        )
        del master_audio

        jobs[job_id].update({
            "progress": 92,
            "status": "မူရင်းအသံဖျောက်ပြီး နောက်ဆုံး MP4 Video ဖန်တီးနေပါသည်...",
        })
        output_mp4 = job_dir / "output_burmese_dubbed.mp4"
        codec = get_video_codec(str(input_video_path))
        v_args = (
            ["-c:v", "copy"]
            if codec == "h264"
            else ["-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", "-crf", "23"]
        )

        run_cmd([
            "ffmpeg", "-y", "-i", str(input_video_path), "-i", str(dubbed_wav),
            "-map", "0:v:0", "-map", "1:a:0", *v_args,
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", str(output_mp4),
        ])

        jobs[job_id].update({
            "progress": 100,
            "status": "ပြီးစီးပါပြီ! MP4 Download လုပ်နိုင်ပါပြီ။",
            "done": True,
            "output_file": str(output_mp4),
        })
    except Exception as e:
        jobs[job_id].update({"error": str(e), "status": f"အမှားဖြစ်ပေါ်ခဲ့ပါသည်: {e}", "done": True})


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
async def get_config():
    return {"has_server_key": bool(SERVER_API_KEY)}


@app.post("/api/dub")
async def start_dubbing(
    video: UploadFile = File(...),
    source_lang: str = Form("auto"),
    voice: str = Form("my-MM-NilarNeural"),
    api_key: str = Form(""),
):
    effective_key = api_key.strip() or SERVER_API_KEY
    if not effective_key:
        raise HTTPException(status_code=400, detail="Gemini API Key ထည့်သွင်းပေးပါ။")

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(video.filename or "input.mp4").suffix or ".mp4"
    input_path = job_dir / f"input{ext}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    jobs[job_id] = {
        "job_id": job_id, "progress": 2,
        "status": "Video Upload ပြီးပါပြီ။ စတင်လုပ်ဆောင်နေပါသည်...",
        "done": False, "error": None, "segments": [], "detected_lang": None, "output_file": None,
    }
    threading.Thread(
        target=process_video_pipeline,
        args=(job_id, input_path, source_lang, voice, effective_key),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "job_id": job["job_id"], "progress": job["progress"], "status": job["status"],
        "done": job["done"], "error": job["error"], "detected_lang": job["detected_lang"],
        "segments": [
            {"start": round(s["start"], 2), "end": round(s["end"], 2), "orig_text": s["orig_text"], "my_text": s["my_text"]}
            for s in job["segments"]
        ],
    }


@app.get("/api/preview/{job_id}")
async def preview_video(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.get("output_file") or not os.path.exists(job["output_file"]):
        raise HTTPException(status_code=404, detail="Output video not ready")
    return FileResponse(job["output_file"], media_type="video/mp4")


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.get("output_file") or not os.path.exists(job["output_file"]):
        raise HTTPException(status_code=404, detail="Output video not ready")
    return FileResponse(job["output_file"], media_type="video/mp4", filename=f"burmese_dubbed_{job_id[:8]}.mp4")
