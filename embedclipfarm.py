#!/usr/bin/env python3
"""
EmbedClipFarm — Semantic search over YouTube content.

Commands:
  index       Index YouTube videos (transcripts + optional vision/whisper)
  search      Search indexed videos by meaning
  shell       Interactive search (loads model once)
  merge       Merge multiple project indexes into one
  clip        Download a specific video clip
  export      Export index as JSON for web UI
  transcripts View or save transcripts
  serve       Local server for web UI clip downloads
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional — env vars still work

# ---------------------------------------------------------------------------
# Lazy imports — heavy libs only loaded when needed
# ---------------------------------------------------------------------------

def _import_chromadb():
    import chromadb
    return chromadb

def _import_sentence_transformers():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

def _import_transcript_api():
    from youtube_transcript_api import YouTubeTranscriptApi
    return YouTubeTranscriptApi

def _import_youtube_api():
    from googleapiclient.discovery import build
    return build

def _import_clip():
    import open_clip
    return open_clip

def _import_pil():
    from PIL import Image
    return Image

def _import_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel

import numpy as np


# ---------------------------------------------------------------------------
# YouTube helpers
# ---------------------------------------------------------------------------

VIDEO_ID_RE = re.compile(r'(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([a-zA-Z0-9_-]{11})')
BARE_ID_RE = re.compile(r'^[a-zA-Z0-9_-]{11}$')
PLAYLIST_RE = re.compile(r'list=([a-zA-Z0-9_-]+)')
CHANNEL_RE = re.compile(r'youtube\.com/(?:@([\w.-]+)|channel/([\w-]+))')


def extract_video_id(text):
    text = text.strip()
    m = VIDEO_ID_RE.search(text)
    if m:
        return m.group(1)
    if BARE_ID_RE.match(text):
        return text
    return None


def resolve_source(source, api_key=None, max_results=200):
    """Resolve a source string into a list of video IDs."""
    # File with URLs/IDs (txt, csv, json)
    if os.path.isfile(source):
        ext = source.lower().rsplit(".", 1)[-1] if "." in source else ""
        ids = []

        if ext == "json":
            with open(source) as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("videos", data.get("urls", []))
            for item in items:
                text = item if isinstance(item, str) else (
                    item.get("url") or item.get("id") or item.get("video_id") or "")
                vid = extract_video_id(str(text))
                if vid and vid not in ids:
                    ids.append(vid)
        elif ext == "csv":
            with open(source) as f:
                for line in f:
                    for cell in line.split(","):
                        vid = extract_video_id(cell.strip().strip('"').strip("'"))
                        if vid and vid not in ids:
                            ids.append(vid)
        else:
            with open(source) as f:
                for line in f:
                    vid = extract_video_id(line.strip())
                    if vid and vid not in ids:
                        ids.append(vid)

        print(f"  Loaded {len(ids)} video(s) from {source}")
        return ids

    # Single video
    vid = extract_video_id(source)
    if vid:
        return [vid]

    # Playlist
    pm = PLAYLIST_RE.search(source)
    if pm:
        return _fetch_playlist(pm.group(1), api_key, max_results)

    # Channel
    cm = CHANNEL_RE.search(source)
    if cm:
        handle = cm.group(1)
        channel_id = cm.group(2)
        return _fetch_channel(handle=handle, channel_id=channel_id,
                              api_key=api_key, max_results=max_results)

    # Maybe it's a bare channel handle
    if source.startswith("@"):
        return _fetch_channel(handle=source[1:], api_key=api_key, max_results=max_results)

    print(f"Could not parse source: {source}")
    return []


def _fetch_playlist(playlist_id, api_key, max_results):
    if not api_key:
        print("YouTube Data API key required for playlist indexing. Set YOUTUBE_API_KEY in .env or use --api-key.")
        sys.exit(1)
    build = _import_youtube_api()
    yt = build("youtube", "v3", developerKey=api_key)
    ids = []
    token = None
    while len(ids) < max_results:
        resp = yt.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=min(50, max_results - len(ids)),
            pageToken=token,
        ).execute()
        for item in resp.get("items", []):
            vid = item["contentDetails"]["videoId"]
            if vid not in ids:
                ids.append(vid)
        token = resp.get("nextPageToken")
        if not token:
            break
    return ids


def _fetch_channel(handle=None, channel_id=None, api_key=None, max_results=200):
    if not api_key:
        print("YouTube Data API key required for channel indexing. Set YOUTUBE_API_KEY in .env or use --api-key.")
        sys.exit(1)
    build = _import_youtube_api()
    yt = build("youtube", "v3", developerKey=api_key)

    if handle and not channel_id:
        resp = yt.search().list(
            part="snippet", q=handle, type="channel", maxResults=1
        ).execute()
        items = resp.get("items", [])
        if not items:
            print(f"Channel @{handle} not found.")
            return []
        channel_id = items[0]["snippet"]["channelId"]

    resp = yt.channels().list(part="contentDetails", id=channel_id).execute()
    items = resp.get("items", [])
    if not items:
        print(f"Channel {channel_id} not found.")
        return []
    uploads_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    return _fetch_playlist(uploads_id, api_key, max_results)


def fetch_metadata(video_ids, api_key=None):
    """Fetch video metadata. Uses API if key provided, else yt-dlp."""
    metadata = {}
    if api_key:
        build = _import_youtube_api()
        yt = build("youtube", "v3", developerKey=api_key)
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            resp = yt.videos().list(
                part="snippet,contentDetails",
                id=",".join(batch),
            ).execute()
            for item in resp.get("items", []):
                vid = item["id"]
                snip = item["snippet"]
                metadata[vid] = {
                    "title": snip.get("title", ""),
                    "description": snip.get("description", ""),
                    "channel": snip.get("channelTitle", ""),
                    "tags": snip.get("tags", []),
                    "published": snip.get("publishedAt", ""),
                    "thumbnail": snip.get("thumbnails", {}).get("high", {}).get("url", ""),
                }
    else:
        for vid in video_ids:
            meta = _fetch_metadata_ytdlp(vid)
            metadata[vid] = meta if meta else {
                "title": vid,
                "description": "",
                "channel": "",
                "tags": [],
                "published": "",
                "thumbnail": f"https://img.youtube.com/vi/{vid}/hqdefault.jpg",
            }
    return metadata


def _fetch_metadata_ytdlp(video_id):
    """Fetch video metadata using yt-dlp (no API key needed)."""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        result = subprocess.run(
            ["yt-dlp", "--remote-components", "ejs:github", "--dump-json", "--skip-download", url],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        import json as _json
        data = _json.loads(result.stdout)
        return {
            "title": data.get("title", video_id),
            "description": data.get("description", ""),
            "channel": data.get("channel", data.get("uploader", "")),
            "tags": data.get("tags", []),
            "published": data.get("upload_date", ""),
            "thumbnail": data.get("thumbnail", f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

def _punctuate_segments(segments):
    """Add punctuation to transcript segments."""
    try:
        from deepmultilingualpunctuation import PunctuationModel
    except ImportError:
        print("    [punctuate] Install: pip install deepmultilingualpunctuation")
        return segments

    print("    [punctuate] Loading punctuation model...")
    model = PunctuationModel()

    for seg in segments:
        text = seg["text"].strip()
        if text and not text[-1] in ".!?,;:":
            try:
                seg["text"] = model.restore_punctuation(text)
            except Exception:
                pass
    print("    [punctuate] Done.")
    return segments


def fetch_transcripts(video_ids, use_whisper=False, whisper_model_name="base",
                      cookies_browser=None, speaker_id=False, punctuate=True):
    """Fetch transcripts. Tries yt-dlp first, then API, then Whisper."""
    transcripts = {}
    failed = []

    for vid in video_ids:
        segments = _fetch_transcript_ytdlp(vid, cookies_browser=cookies_browser)
        if segments:
            transcripts[vid] = segments
            print(f"  [yt-dlp] {vid}: {len(segments)} segments")
            continue

        try:
            TranscriptApi = _import_transcript_api()
            segs = TranscriptApi.get_transcript(vid)
            transcripts[vid] = [
                {"start": s["start"], "duration": s["duration"], "text": s["text"]}
                for s in segs
            ]
            print(f"  [transcript-api] {vid}: {len(segs)} segments")
        except Exception:
            failed.append(vid)
            print(f"  [transcript] {vid}: no captions via yt-dlp or API")

    if use_whisper and failed:
        print(f"\nRunning Whisper on {len(failed)} video(s) without captions...")
        WhisperModel = _import_whisper()
        model = WhisperModel(whisper_model_name, device="cpu", compute_type="int8")

        for vid in failed:
            try:
                segments = _whisper_transcribe(vid, model,
                                               cookies_browser=cookies_browser,
                                               speaker_id=speaker_id)
                if segments:
                    transcripts[vid] = segments
                    print(f"  [whisper] {vid}: {len(segments)} segments")
                else:
                    print(f"  [whisper] {vid}: transcription failed")
            except Exception as e:
                print(f"  [whisper] {vid}: error - {e}")

    if punctuate and transcripts:
        print("\nAdding punctuation...")
        for vid, segments in transcripts.items():
            transcripts[vid] = _punctuate_segments(segments)

    return transcripts


def _fetch_transcript_ytdlp(video_id, cookies_browser=None):
    """Fetch transcript using yt-dlp."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            url = f"https://www.youtube.com/watch?v={video_id}"
            out_path = os.path.join(tmpdir, "sub")
            cmd = ["yt-dlp", "--remote-components", "ejs:github",
                   "--write-auto-sub", "--sub-lang", "en",
                   "--sub-format", "json3", "--skip-download",
                   "-o", out_path, url]
            if cookies_browser:
                cmd.insert(1, f"--cookies-from-browser={cookies_browser}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            sub_file = None
            for f in Path(tmpdir).glob("*.json3"):
                sub_file = f
                break
            if not sub_file or not sub_file.exists():
                return None

            with open(sub_file) as f:
                data = json.load(f)

            segments = []
            for event in data.get("events", []):
                if "segs" not in event:
                    continue
                text = "".join(s.get("utf8", "") for s in event["segs"]).strip()
                if not text:
                    continue
                segments.append({
                    "start": event.get("tStartMs", 0) / 1000,
                    "duration": event.get("dDurationMs", 0) / 1000,
                    "text": text,
                })
            return segments if segments else None
    except Exception:
        return None


def _diarize(audio_path):
    """Run speaker diarization."""
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                             use_auth_token=os.environ.get("HF_TOKEN"))
        diarization = pipeline(audio_path)
        speaker_map = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_map[(turn.start, turn.end)] = speaker
        return speaker_map
    except ImportError:
        print("    [speakers] Install pyannote-audio for speaker ID: pip install pyannote-audio")
        print("    [speakers] Also set HF_TOKEN in .env (required by pyannote)")
        return {}


def _get_speaker(seg_start, seg_end, speaker_map):
    """Find the dominant speaker for a segment."""
    best_speaker = None
    best_overlap = 0
    for (start, end), speaker in speaker_map.items():
        overlap = max(0, min(seg_end, end) - max(seg_start, start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


def _whisper_transcribe(video_id, model, cookies_browser=None, speaker_id=False):
    """Download audio via yt-dlp and transcribe with Whisper."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        url = f"https://www.youtube.com/watch?v={video_id}"
        cmd = ["yt-dlp", "--remote-components", "ejs:github",
               "-x", "--audio-format", "wav",
               "--postprocessor-args", "-ac 1 -ar 16000",
               "-o", audio_path, url]
        if cookies_browser:
            cmd.insert(1, f"--cookies-from-browser={cookies_browser}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if not os.path.exists(audio_path):
            for f in Path(tmpdir).glob("audio.*"):
                audio_path = str(f)
                break
        if not os.path.exists(audio_path):
            return None

        speaker_map = {}
        if speaker_id:
            try:
                speaker_map = _diarize(audio_path)
                print(f"    [speakers] found {len(set(speaker_map.values()))} speaker(s)")
            except Exception as e:
                print(f"    [speakers] diarization failed: {e}")

        segments_iter, _ = model.transcribe(audio_path, beam_size=5)
        segments = []
        for seg in segments_iter:
            text = seg.text.strip()
            if speaker_map:
                speaker = _get_speaker(seg.start, seg.end, speaker_map)
                if speaker:
                    text = f"[{speaker}] {text}"
            segments.append({
                "start": seg.start,
                "duration": seg.end - seg.start,
                "text": text,
            })
        return segments


# ---------------------------------------------------------------------------
# Text chunking & embedding
# ---------------------------------------------------------------------------

def chunk_transcript(segments, chunk_seconds=30):
    """Group transcript segments into ~N-second chunks."""
    chunks = []
    current_texts = []
    current_start = None
    current_end = 0

    for seg in segments:
        if current_start is None:
            current_start = seg["start"]
        current_texts.append(seg["text"])
        current_end = seg["start"] + seg["duration"]

        if current_end - current_start >= chunk_seconds:
            chunks.append({
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_texts),
            })
            current_texts = []
            current_start = None

    if current_texts:
        chunks.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_texts),
        })

    return chunks


def embed_texts(texts, model):
    """Embed a list of texts using sentence-transformers model."""
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# CLIP keyframe extraction & embedding
# ---------------------------------------------------------------------------

def extract_keyframes(video_id, interval=30, max_frames=20, cookies_browser=None):
    """Extract keyframes from a YouTube video via yt-dlp streaming + ffmpeg."""
    frames = []
    Image = _import_pil()

    with tempfile.TemporaryDirectory() as tmpdir:
        url = f"https://www.youtube.com/watch?v={video_id}"
        cmd = ["yt-dlp", "--remote-components", "ejs:github", "-f", "best[height<=720]", "-g", url]
        if cookies_browser:
            cmd.insert(1, f"--cookies-from-browser={cookies_browser}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        stream_url = result.stdout.strip().split("\n")[0]
        if not stream_url:
            print(f"  [keyframes] {video_id}: could not get stream URL")
            return []

        pattern = os.path.join(tmpdir, "frame_%04d.jpg")
        subprocess.run(
            ["ffmpeg", "-i", stream_url,
             "-vf", f"fps=1/{interval}",
             "-frames:v", str(max_frames),
             "-q:v", "2",
             pattern],
            capture_output=True, text=True
        )

        for fpath in sorted(Path(tmpdir).glob("frame_*.jpg")):
            idx = int(fpath.stem.split("_")[1]) - 1
            timestamp = idx * interval
            img = Image.open(fpath).convert("RGB")
            frames.append({"timestamp": timestamp, "image": img, "path": str(fpath)})

        print(f"  [keyframes] {video_id}: {len(frames)} frames extracted")

    return frames


def embed_keyframes(frames, clip_model, preprocess, tokenizer):
    """Embed keyframe images using CLIP."""
    import torch
    embeddings = []
    for frame in frames:
        img_tensor = preprocess(frame["image"]).unsqueeze(0)
        with torch.no_grad():
            emb = clip_model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.squeeze().cpu().numpy().tolist())
    return embeddings


# ---------------------------------------------------------------------------
# Vision API scene annotation (Gemini / Claude)
# ---------------------------------------------------------------------------

SCENE_PROMPT = (
    "Describe this video frame in detail for search indexing. "
    "Include: what's happening, who/what is visible, the setting, "
    "any text on screen, emotions, actions, and visual style. "
    "Be specific and descriptive in 2-3 sentences."
)

QA_PROMPT = (
    "Answer each question about this video frame concisely:\n"
    "1. OBJECTS: What objects, items, and props are visible?\n"
    "2. TEXT: What text, titles, captions, or graphics appear on screen?\n"
    "3. PEOPLE: Describe any people — appearance, clothing, actions, expressions.\n"
    "4. SETTING: What is the location or environment?\n"
    "5. ACTION: What is happening in this moment?\n"
    "Format each answer on its own line starting with the label."
)


def _frame_to_base64(frame):
    """Convert a PIL image to base64 JPEG."""
    import base64
    import io
    buf = io.BytesIO()
    frame["image"].save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def annotate_keyframes(frames, provider, api_key, prompt=None):
    """Annotate keyframes using the specified vision API provider."""
    use_prompt = prompt or SCENE_PROMPT
    if provider == "gemini":
        return _annotate_gemini(frames, api_key, use_prompt)
    elif provider == "claude":
        return _annotate_claude(frames, api_key, use_prompt)
    else:
        raise ValueError(f"Unknown vision provider: {provider}")


def _annotate_gemini(frames, api_key, prompt):
    """Use Gemini to generate descriptions of keyframes."""
    import urllib.request

    descriptions = []
    for frame in frames:
        img_b64 = _frame_to_base64(frame)
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                ]
            }]
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            descriptions.append(text)
            print(f"    [{_fmt(frame['timestamp'])}] {text[:100]}...")
        except Exception as e:
            descriptions.append(f"Video frame at {_fmt(frame['timestamp'])}")
            print(f"    [{_fmt(frame['timestamp'])}] Gemini error: {e}")

    return descriptions


def _annotate_claude(frames, api_key, prompt):
    """Use Claude to generate descriptions of keyframes."""
    import urllib.request

    descriptions = []
    for frame in frames:
        img_b64 = _frame_to_base64(frame)
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 300,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        url = "https://api.anthropic.com/v1/messages"

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            text = data["content"][0]["text"].strip()
            descriptions.append(text)
            print(f"    [{_fmt(frame['timestamp'])}] {text[:100]}...")
        except Exception as e:
            descriptions.append(f"Video frame at {_fmt(frame['timestamp'])}")
            print(f"    [{_fmt(frame['timestamp'])}] Claude error: {e}")

    return descriptions


# ---------------------------------------------------------------------------
# Fused chunks (transcript + vision at same timestamp)
# ---------------------------------------------------------------------------

def fuse_chunks_with_vision(chunks, vision_descriptions, frame_interval):
    """Merge transcript chunks with vision descriptions at overlapping timestamps."""
    fused = []
    for chunk in chunks:
        # Find vision descriptions that overlap with this chunk's time range
        visual_parts = []
        for desc in vision_descriptions:
            if desc["end"] > chunk["start"] and desc["start"] < chunk["end"]:
                visual_parts.append(desc["text"])

        if visual_parts:
            fused_text = (
                f"TRANSCRIPT: {chunk['text']}\n"
                f"VISUAL: {' '.join(visual_parts)}"
            )
        else:
            fused_text = chunk["text"]

        fused.append({
            "start": chunk["start"],
            "end": chunk["end"],
            "text": fused_text,
            "type": "fused" if visual_parts else "text",
        })

    return fused


# ---------------------------------------------------------------------------
# ChromaDB storage
# ---------------------------------------------------------------------------

class VectorStore:
    def __init__(self, db_path="./embedclipfarm_db"):
        chromadb = _import_chromadb()
        self.client = chromadb.PersistentClient(path=db_path)
        self.text_collection = self.client.get_or_create_collection(
            name="text_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self.visual_collection = self.client.get_or_create_collection(
            name="visual_frames",
            metadata={"hnsw:space": "cosine"},
        )
        self.meta_collection = self.client.get_or_create_collection(
            name="video_metadata",
        )

    def has_video(self, video_id):
        """Check if a video is already indexed."""
        result = self.meta_collection.get(ids=[video_id])
        return len(result["ids"]) > 0

    def add_metadata(self, video_id, metadata):
        self.meta_collection.upsert(
            ids=[video_id],
            documents=[json.dumps(metadata)],
            metadatas=[{"video_id": video_id, "title": metadata.get("title", "")}],
        )

    def add_text_chunks(self, video_id, chunks, embeddings):
        ids = [f"{video_id}_t_{i}" for i in range(len(chunks))]
        docs = [c["text"] for c in chunks]
        metas = [
            {
                "video_id": video_id,
                "start": c["start"],
                "end": c["end"],
                "type": c.get("type", "text"),
            }
            for c in chunks
        ]
        self.text_collection.upsert(
            ids=ids, documents=docs, embeddings=embeddings, metadatas=metas,
        )

    def add_visual_frames(self, video_id, frames, embeddings):
        ids = [f"{video_id}_v_{i}" for i in range(len(frames))]
        docs = [f"Frame at {f['timestamp']}s" for f in frames]
        metas = [
            {
                "video_id": video_id,
                "timestamp": f["timestamp"],
                "type": "visual",
            }
            for f in frames
        ]
        self.visual_collection.upsert(
            ids=ids, documents=docs, embeddings=embeddings, metadatas=metas,
        )

    def search_text(self, query_embedding, top_k=10, where=None):
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = self.text_collection.query(**kwargs)
        return self._format_results(results)

    def search_visual(self, query_embedding, top_k=10):
        results = self.visual_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return self._format_results(results)

    def get_metadata(self, video_id):
        result = self.meta_collection.get(ids=[video_id])
        if result["documents"]:
            return json.loads(result["documents"][0])
        return {}

    def get_all_metadata(self):
        result = self.meta_collection.get()
        out = {}
        for doc_id, doc in zip(result["ids"], result["documents"]):
            out[doc_id] = json.loads(doc)
        return out

    def export_all(self):
        """Export entire DB as a dict for JSON serialization."""
        text_data = self.text_collection.get(include=["embeddings", "documents", "metadatas"])
        visual_data = self.visual_collection.get(include=["embeddings", "documents", "metadatas"])
        meta_data = self.meta_collection.get(include=["documents", "metadatas"])

        return {
            "text_chunks": {
                "ids": text_data["ids"],
                "embeddings": text_data["embeddings"],
                "documents": text_data["documents"],
                "metadatas": text_data["metadatas"],
            },
            "visual_frames": {
                "ids": visual_data["ids"],
                "embeddings": visual_data["embeddings"],
                "documents": visual_data["documents"],
                "metadatas": visual_data["metadatas"],
            },
            "videos": {
                vid: json.loads(doc)
                for vid, doc in zip(meta_data["ids"], meta_data["documents"])
            },
        }

    @staticmethod
    def _format_results(results):
        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],
            })
        return items


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def _derive_project_name(sources):
    """Derive a folder name from the source(s)."""
    names = []
    for source in sources:
        s = source.strip()
        if s.startswith("@"):
            names.append(s[1:])
        elif os.path.isfile(s):
            names.append(Path(s).stem)
        else:
            vid = extract_video_id(s)
            if vid:
                names.append(vid)
            else:
                m = re.search(r'list=([a-zA-Z0-9_-]+)', s)
                if m:
                    names.append(m.group(1)[:20])
                else:
                    m = re.search(r'@([\w.-]+)', s)
                    if m:
                        names.append(m.group(1))
                    else:
                        names.append("index")
    return "_".join(names)[:80] if names else "index"


def _make_serializable(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    return obj


def cmd_index(args):
    video_ids = []
    for source in args.source:
        print(f"Resolving source: {source}")
        ids = resolve_source(source, api_key=args.api_key,
                             max_results=args.max_videos)
        for vid in ids:
            if vid not in video_ids:
                video_ids.append(vid)
    if not video_ids:
        print("No video IDs found.")
        return
    print(f"Found {len(video_ids)} video(s).\n")

    # Auto-derive output folder
    if args.db_path == "./embedclipfarm_db":
        project_name = _derive_project_name(args.source)
        project_dir = os.path.join(".", project_name)
        db_path = os.path.join(project_dir, "db")
    else:
        project_dir = os.path.dirname(args.db_path) or "."
        db_path = args.db_path

    os.makedirs(project_dir, exist_ok=True)

    store = VectorStore(db_path=db_path)

    # Incremental indexing — skip already-indexed videos
    new_ids = []
    skipped = 0
    for vid in video_ids:
        if store.has_video(vid):
            skipped += 1
        else:
            new_ids.append(vid)
    if skipped:
        print(f"Skipping {skipped} already-indexed video(s).")
    if not new_ids:
        print("All videos already indexed. Nothing to do.")
        print(f"Output folder: {project_dir}/")
        return
    video_ids = new_ids
    print(f"Indexing {len(video_ids)} new video(s).\n")

    # Metadata
    print("Fetching metadata...")
    metadata = fetch_metadata(video_ids, api_key=args.api_key)

    # Transcripts
    print("\nFetching transcripts...")
    transcripts = fetch_transcripts(video_ids, use_whisper=args.whisper,
                                     whisper_model_name=args.whisper_model,
                                     cookies_browser=args.cookies_from_browser,
                                     speaker_id=args.speaker_id,
                                     punctuate=not args.no_punctuate)

    # Text embeddings
    print("\nLoading text embedding model...")
    SentenceTransformer = _import_sentence_transformers()
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Vision annotation (collect descriptions per video for fusing)
    vision_descs = {}  # {video_id: [{start, end, text}, ...]}
    if args.vision:
        provider = args.vision
        if provider == "gemini":
            vision_key = args.vision_key or os.environ.get("GEMINI_API_KEY")
            key_name = "GEMINI_API_KEY"
        elif provider == "claude":
            vision_key = args.vision_key or os.environ.get("ANTHROPIC_API_KEY")
            key_name = "ANTHROPIC_API_KEY"
        else:
            vision_key = None
            key_name = ""

        if not vision_key:
            print(f"\n{provider.capitalize()} API key required. Set {key_name} in .env or use --vision-key.")
        else:
            # Use QA prompt if --vision-qa flag is set
            prompt = args.vision_prompt
            if args.vision_qa and not args.vision_prompt:
                prompt = QA_PROMPT

            print(f"\nExtracting keyframes and annotating with {provider.capitalize()}...")
            for vid in video_ids:
                frames = extract_keyframes(vid, interval=args.frame_interval,
                                            max_frames=args.max_frames,
                                            cookies_browser=args.cookies_from_browser)
                if frames:
                    meta = metadata.get(vid, {})
                    print(f"  [{provider}] {vid} — {len(frames)} frames — {meta.get('title', '')[:40]}")
                    descriptions = annotate_keyframes(frames, provider, vision_key, prompt=prompt)
                    vision_descs[vid] = [
                        {"start": f["timestamp"], "end": f["timestamp"] + args.frame_interval, "text": d}
                        for f, d in zip(frames, descriptions)
                    ]

    # Index transcripts (with optional fusion)
    print("\nIndexing transcripts...")
    for vid in video_ids:
        meta = metadata.get(vid, {})
        store.add_metadata(vid, meta)

        if vid in transcripts:
            chunks = chunk_transcript(transcripts[vid], chunk_seconds=args.chunk_seconds)
            if chunks:
                # Fuse with vision descriptions if available
                if vid in vision_descs:
                    chunks = fuse_chunks_with_vision(chunks, vision_descs[vid], args.frame_interval)
                    print(f"  [fused] {vid}: {len(chunks)} fused chunks (transcript + vision)")
                else:
                    for c in chunks:
                        c["type"] = "text"

                texts = [c["text"] for c in chunks]
                embeddings = embed_texts(texts, text_model)
                store.add_text_chunks(vid, chunks, embeddings)
                if vid not in vision_descs:
                    print(f"  [indexed] {vid}: {len(chunks)} text chunks"
                          f" — {meta.get('title', '')[:50]}")

                if args.show_transcripts:
                    for c in chunks:
                        print(f"    [{_fmt(c['start'])} → {_fmt(c['end'])}] {c['text'][:200]}")
        else:
            # Index vision-only descriptions if no transcript
            if vid in vision_descs:
                descs = vision_descs[vid]
                chunks = [{"start": d["start"], "end": d["end"], "text": d["text"], "type": args.vision} for d in descs]
                texts = [d["text"] for d in descs]
                embeddings = embed_texts(texts, text_model)
                store.add_text_chunks(vid, chunks, embeddings)
                print(f"  [indexed] {vid}: {len(descs)} {args.vision} descriptions (no transcript)")
            else:
                print(f"  [skipped] {vid}: no transcript or vision data")

    # Always save transcripts
    transcript_dir = os.path.join(project_dir, "transcripts")
    os.makedirs(transcript_dir, exist_ok=True)
    saved = 0
    for vid in video_ids:
        if vid not in transcripts:
            continue
        meta = metadata.get(vid, {})
        title = meta.get("title", vid)
        safe = re.sub(r'[^\w\s-]', '', title)[:60].strip().replace(' ', '_')
        filepath = os.path.join(transcript_dir, f"{safe}_{vid}.txt")
        segs = transcripts[vid]
        with open(filepath, "w") as f:
            f.write(f"# {title}\n")
            f.write(f"# https://youtube.com/watch?v={vid}\n\n")
            for s in segs:
                f.write(f"[{_fmt(s['start'])}] {s['text']}\n")
        saved += 1
    print(f"\n{saved} transcript(s) saved to {transcript_dir}/")

    # CLIP keyframes (separate visual collection)
    if args.clip:
        print("\nLoading CLIP model...")
        open_clip = _import_clip()
        import torch
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        clip_model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        print("\nExtracting and embedding keyframes with CLIP...")
        for vid in video_ids:
            frames = extract_keyframes(vid, interval=args.frame_interval,
                                        max_frames=args.max_frames,
                                        cookies_browser=args.cookies_from_browser)
            if frames:
                embeddings = embed_keyframes(frames, clip_model, preprocess, tokenizer)
                store.add_visual_frames(vid, frames, embeddings)
                print(f"  [indexed] {vid}: {len(frames)} CLIP visual frames")

    text_count = store.text_collection.count()
    visual_count = store.visual_collection.count()

    # Auto-export index.json
    index_path = os.path.join(project_dir, "index.json")
    data = _make_serializable(store.export_all())
    with open(index_path, "w") as f:
        json.dump(data, f)
    index_mb = os.path.getsize(index_path) / 1048576

    print(f"\nDone! Indexed {len(video_ids)} videos,"
          f" {text_count} text chunks, {visual_count} visual frames.")
    print(f"\nOutput folder: {project_dir}/")
    print(f"  index.json     ({index_mb:.1f} MB) — load in web UI to search")
    print(f"  transcripts/   — raw transcript text files")
    print(f"  db/            — ChromaDB vector database")
    print(f"\nTo search: python embedclipfarm.py search \"your query\" --db-path {db_path}")
    print(f"Web UI:    open index.html → load {index_path}")


def _find_db_path(db_path):
    """Auto-find a database. Prefers project subfolders over the default."""
    dbs = sorted(Path(".").glob("*/db/chroma.sqlite3"))
    if len(dbs) == 1:
        found = str(dbs[0].parent)
        print(f"Auto-detected database: {found}")
        return found
    elif len(dbs) > 1:
        for d in dbs:
            if str(d.parent) == db_path or db_path in str(d.parent):
                return str(d.parent)
        print("Multiple databases found. Specify one with --db-path:")
        for d in dbs:
            print(f"  --db-path {d.parent}")
        sys.exit(1)

    if os.path.exists(db_path):
        return db_path

    print(f"No database found. Run 'index' first or specify --db-path.")
    sys.exit(1)


def _find_clips_dir(explicit_dir):
    """Auto-detect the clips directory inside a project subfolder."""
    if explicit_dir != "./clips":
        return explicit_dir
    dbs = sorted(Path(".").glob("*/db/chroma.sqlite3"))
    if len(dbs) == 1:
        return str(dbs[0].parent.parent / "clips")
    return explicit_dir


def cmd_search_and_clip(args):
    """Search and optionally download matching clips."""
    args.db_path = _find_db_path(args.db_path)
    store = VectorStore(db_path=args.db_path)

    SentenceTransformer = _import_sentence_transformers()
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = text_model.encode([args.query], normalize_embeddings=True).tolist()[0]

    # Build filter
    where = None
    if hasattr(args, 'filter_video') and args.filter_video:
        where = {"video_id": args.filter_video}
    elif hasattr(args, 'filter_type') and args.filter_type:
        where = {"type": args.filter_type}

    results = store.search_text(query_emb, top_k=args.top_k, where=where)

    # Filter by minimum score
    min_score = getattr(args, 'min_score', 0)
    if min_score:
        results = [r for r in results if r["score"] >= min_score]

    print(f"\n{'='*60}")
    print(f"TEXT SEARCH: \"{args.query}\"")
    print(f"{'='*60}")

    clips_to_download = []
    for i, r in enumerate(results):
        meta = r["metadata"]
        vid_meta = store.get_metadata(meta["video_id"])
        title = vid_meta.get("title", meta["video_id"])
        start = meta.get("start", 0)
        end = meta.get("end", 0)
        source_type = meta.get("type", "text")
        print(f"\n  [{i+1}] [{r['score']:.3f}] [{source_type}] {title}")
        print(f"         Video: https://youtube.com/watch?v={meta['video_id']}&t={int(start)}s")
        print(f"         Time:  {_fmt(start)} → {_fmt(end)}")
        print(f"         Text:  {r['text'][:150]}...")
        print(f"         Clip:  python embedclipfarm.py clip {meta['video_id']} --start {start:.0f} --end {end:.0f}")
        clips_to_download.append((meta["video_id"], start, end, title))

    # SRT export
    if hasattr(args, 'srt') and args.srt:
        _export_srt(results, store, args.srt)

    # Compile video
    if hasattr(args, 'compile') and args.compile:
        _compile_clips(clips_to_download, args.compile, getattr(args, 'cookies_from_browser', None))

    # Download clips
    if args.download:
        out_dir = _find_clips_dir(args.download) if args.download == "./clips" else args.download
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Downloading {len(clips_to_download)} clip(s) to {out_dir}/...")
        print(f"{'='*60}")
        for vid, start, end, title in clips_to_download:
            actual_start = max(0, start - 2)
            actual_end = end + 2
            safe_title = re.sub(r'[^\w\s-]', '', title)[:40].strip().replace(' ', '_')
            filename = f"{safe_title}_{int(start)}s-{int(end)}s.mp4"
            output_path = os.path.join(out_dir, filename)
            url = f"https://www.youtube.com/watch?v={vid}"

            print(f"\n  Downloading: {title} [{_fmt(start)} → {_fmt(end)}]")
            cmd = [
                "yt-dlp", "--remote-components", "ejs:github",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--download-sections", f"*{actual_start}-{actual_end}",
                "--force-keyframes-at-cuts",
                "-o", output_path,
                url,
            ]
            if getattr(args, 'cookies_from_browser', None):
                cmd.insert(1, f"--cookies-from-browser={args.cookies_from_browser}")
            subprocess.run(cmd, capture_output=True)
            if os.path.exists(output_path):
                size = os.path.getsize(output_path) / 1048576
                print(f"  Saved: {output_path} ({size:.1f} MB)")
            else:
                print(f"  Failed to download clip")

        print(f"\nDone! Clips saved to {out_dir}/")


def _export_srt(results, store, output_path):
    """Export search results as SRT subtitle file."""
    with open(output_path, "w") as f:
        for i, r in enumerate(results):
            meta = r["metadata"]
            start = meta.get("start", 0)
            end = meta.get("end", start + 30)
            f.write(f"{i+1}\n")
            f.write(f"{_fmt_srt(start)} --> {_fmt_srt(end)}\n")
            f.write(f"{r['text'][:200]}\n\n")
    print(f"\nSRT exported to {output_path}")


def _compile_clips(clips, output_path, cookies_browser=None):
    """Download all clips and stitch them into one video."""
    with tempfile.TemporaryDirectory() as tmpdir:
        clip_files = []
        for i, (vid, start, end, title) in enumerate(clips):
            actual_start = max(0, start - 1)
            actual_end = end + 1
            clip_path = os.path.join(tmpdir, f"clip_{i:03d}.mp4")
            url = f"https://www.youtube.com/watch?v={vid}"
            cmd = [
                "yt-dlp", "--remote-components", "ejs:github",
                "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--download-sections", f"*{actual_start}-{actual_end}",
                "--force-keyframes-at-cuts",
                "-o", clip_path,
                url,
            ]
            if cookies_browser:
                cmd.insert(1, f"--cookies-from-browser={cookies_browser}")

            print(f"  [{i+1}/{len(clips)}] Downloading: {title[:40]} [{_fmt(start)} → {_fmt(end)}]")
            subprocess.run(cmd, capture_output=True)
            if os.path.exists(clip_path):
                clip_files.append(clip_path)

        if not clip_files:
            print("  No clips downloaded. Compilation skipped.")
            return

        # Write ffmpeg concat list
        list_path = os.path.join(tmpdir, "concat.txt")
        with open(list_path, "w") as f:
            for cf in clip_files:
                f.write(f"file '{cf}'\n")

        print(f"  Stitching {len(clip_files)} clips...")
        subprocess.run(
            ["ffmpeg", "-f", "concat", "-safe", "0", "-i", list_path,
             "-c", "copy", output_path],
            capture_output=True,
        )

        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / 1048576
            print(f"\nCompilation saved: {output_path} ({size:.1f} MB)")
        else:
            print(f"\nCompilation failed.")


def cmd_clip(args):
    """Download a specific clip from a YouTube video."""
    video_id = extract_video_id(args.video) or args.video
    start = args.start
    end = args.end
    padding = args.padding

    actual_start = max(0, start - padding)
    actual_end = end + padding

    url = f"https://www.youtube.com/watch?v={video_id}"
    out_dir = _find_clips_dir(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{video_id}_{int(start)}s-{int(end)}s.mp4"
    output_path = os.path.join(out_dir, filename)

    print(f"Downloading clip: {video_id} [{_fmt(actual_start)} → {_fmt(actual_end)}]")

    cmd = [
        "yt-dlp", "--remote-components", "ejs:github",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--download-sections", f"*{actual_start}-{actual_end}",
        "--force-keyframes-at-cuts",
        "-o", output_path,
        url,
    ]
    if args.cookies_from_browser:
        cmd.insert(1, f"--cookies-from-browser={args.cookies_from_browser}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0 and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1048576
        print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")
    else:
        print(f"\nDownload failed. Try adding --cookies-from-browser chrome")


def cmd_export(args):
    args.db_path = _find_db_path(args.db_path)
    store = VectorStore(db_path=args.db_path)
    data = _make_serializable(store.export_all())
    with open(args.output, "w") as f:
        json.dump(data, f)

    text_count = len(data["text_chunks"]["ids"])
    visual_count = len(data["visual_frames"]["ids"])
    video_count = len(data["videos"])
    size_mb = os.path.getsize(args.output) / 1048576

    print(f"Exported {video_count} videos, {text_count} text chunks,"
          f" {visual_count} visual frames")
    print(f"Output: {args.output} ({size_mb:.1f} MB)")


def cmd_transcripts(args):
    args.db_path = _find_db_path(args.db_path)
    store = VectorStore(db_path=args.db_path)
    data = store.text_collection.get(include=["documents", "metadatas"])
    all_metadata = store.get_all_metadata()

    videos = {}
    for i, meta in enumerate(data["metadatas"]):
        vid = meta["video_id"]
        if args.video and vid != args.video:
            continue
        if vid not in videos:
            videos[vid] = []
        videos[vid].append({
            "start": meta.get("start", 0),
            "end": meta.get("end", 0),
            "text": data["documents"][i],
        })

    if not videos:
        print("No transcripts found." if not args.video else f"No transcript for: {args.video}")
        return

    for vid in videos:
        videos[vid].sort(key=lambda c: c["start"])

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for vid, chunks in videos.items():
            vid_meta = all_metadata.get(vid, {})
            title = vid_meta.get("title", vid)
            safe_name = re.sub(r'[^\w\s-]', '', title)[:60].strip().replace(' ', '_')
            filepath = os.path.join(args.save, f"{safe_name}_{vid}.txt")
            with open(filepath, "w") as f:
                f.write(f"# {title}\n# https://youtube.com/watch?v={vid}\n\n")
                for chunk in chunks:
                    f.write(f"[{_fmt(chunk['start'])} → {_fmt(chunk['end'])}]\n{chunk['text']}\n\n")
            print(f"  Saved: {filepath}")
        print(f"\n{len(videos)} transcript(s) saved to {args.save}/")
        return

    for vid, chunks in videos.items():
        vid_meta = all_metadata.get(vid, {})
        title = vid_meta.get("title", vid)
        print(f"\n{'='*60}\n{title}\nhttps://youtube.com/watch?v={vid}\n{'='*60}")
        for chunk in chunks:
            print(f"\n[{_fmt(chunk['start'])} → {_fmt(chunk['end'])}]\n{chunk['text']}")
    print(f"\n{len(videos)} video(s), {sum(len(c) for c in videos.values())} chunks total.")


def cmd_shell(args):
    """Interactive search shell — loads model once."""
    args.db_path = _find_db_path(args.db_path)
    store = VectorStore(db_path=args.db_path)

    print("Loading embedding model...")
    SentenceTransformer = _import_sentence_transformers()
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    text_count = store.text_collection.count()
    print(f"Ready! {text_count} chunks indexed. Type a query, or 'quit' to exit.\n")

    while True:
        try:
            query = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        query_emb = text_model.encode([query], normalize_embeddings=True).tolist()[0]
        results = store.search_text(query_emb, top_k=args.top_k)

        for i, r in enumerate(results):
            meta = r["metadata"]
            vid_meta = store.get_metadata(meta["video_id"])
            title = vid_meta.get("title", meta["video_id"])
            start = meta.get("start", 0)
            end = meta.get("end", 0)
            src = meta.get("type", "text")
            print(f"\n  [{i+1}] [{r['score']:.3f}] [{src}] {title}")
            print(f"       https://youtube.com/watch?v={meta['video_id']}&t={int(start)}s")
            print(f"       {_fmt(start)} → {_fmt(end)}: {r['text'][:120]}...")
        print()


def cmd_merge(args):
    """Merge multiple project indexes into one."""
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    out_db = os.path.join(output_dir, "db")
    out_store = VectorStore(db_path=out_db)

    total_videos = 0
    total_chunks = 0

    for source_dir in args.sources:
        db_dir = os.path.join(source_dir, "db")
        if not os.path.exists(db_dir):
            db_dir = source_dir  # maybe they passed the db path directly
        if not os.path.exists(os.path.join(db_dir, "chroma.sqlite3")):
            print(f"  Skipping {source_dir} — no database found")
            continue

        src_store = VectorStore(db_path=db_dir)
        src_data = src_store.export_all()

        # Merge metadata
        for vid, meta in src_data["videos"].items():
            out_store.add_metadata(vid, meta)
            total_videos += 1

        # Merge text chunks
        tc = src_data["text_chunks"]
        if tc["ids"]:
            out_store.text_collection.upsert(
                ids=tc["ids"],
                documents=tc["documents"],
                embeddings=[e if isinstance(e, list) else e.tolist() for e in tc["embeddings"]],
                metadatas=tc["metadatas"],
            )
            total_chunks += len(tc["ids"])

        # Merge visual frames
        vc = src_data["visual_frames"]
        if vc["ids"]:
            out_store.visual_collection.upsert(
                ids=vc["ids"],
                documents=vc["documents"],
                embeddings=[e if isinstance(e, list) else e.tolist() for e in vc["embeddings"]],
                metadatas=vc["metadatas"],
            )

        print(f"  Merged: {source_dir} ({len(src_data['videos'])} videos, {len(tc['ids'])} chunks)")

    # Auto-export
    index_path = os.path.join(output_dir, "index.json")
    data = _make_serializable(out_store.export_all())
    with open(index_path, "w") as f:
        json.dump(data, f)

    print(f"\nMerged {total_videos} videos, {total_chunks} chunks into {output_dir}/")
    print(f"  index.json + db/ ready.")


def cmd_serve(args):
    """Run a local server that serves the web UI and handles clip downloads."""
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs

    port = args.port
    serve_dir = os.path.dirname(os.path.abspath(__file__))

    class ClipHandler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=serve_dir, **kw)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/clip":
                self.handle_clip(parse_qs(parsed.query))
            elif parsed.path == "/api/status":
                self.send_json({"status": "ok"})
            else:
                super().do_GET()

        def handle_clip(self, params):
            video_id = params.get("video", [None])[0]
            start = params.get("start", [None])[0]
            end = params.get("end", [None])[0]
            if not video_id or start is None or end is None:
                self.send_json({"error": "Missing video, start, or end"}, 400)
                return

            start_f, end_f = float(start), float(end)
            actual_start = max(0, start_f - 2)
            actual_end = end_f + 2
            url = f"https://www.youtube.com/watch?v={video_id}"

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, f"{video_id}_{int(start_f)}s-{int(end_f)}s.mp4")
                cmd = [
                    "yt-dlp", "--remote-components", "ejs:github",
                    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "--download-sections", f"*{actual_start}-{actual_end}",
                    "--force-keyframes-at-cuts", "-o", output_path, url,
                ]
                if args.cookies_from_browser:
                    cmd.insert(1, f"--cookies-from-browser={args.cookies_from_browser}")

                print(f"  Downloading clip: {video_id} [{_fmt(start_f)} → {_fmt(end_f)}]")
                subprocess.run(cmd, capture_output=True)

                if not os.path.exists(output_path):
                    self.send_json({"error": "Download failed"}, 500)
                    return

                file_size = os.path.getsize(output_path)
                filename = f"{video_id}_{int(start_f)}s-{int(end_f)}s.mp4"
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(file_size))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(output_path, "rb") as f:
                    self.wfile.write(f.read())
                print(f"  Sent: {filename} ({file_size / 1048576:.1f} MB)")

        def send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *a):
            if "/api/" in (a[0] if a else ""):
                super().log_message(fmt, *a)

    server = HTTPServer(("127.0.0.1", port), ClipHandler)
    print(f"\nEmbedClipFarm server running at http://localhost:{port}")
    print(f"Press Ctrl+C to stop.\n")

    import webbrowser
    webbrowser.open(f"http://localhost:{port}/index.html")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


def _fmt(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _fmt_srt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EmbedClipFarm — Semantic search over YouTube content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # --- index ---
    idx = sub.add_parser("index", help="Index YouTube videos")
    idx.add_argument("source", nargs="+",
        help="YouTube URLs, @handles, playlist URLs, or files (txt/csv/json)")
    idx.add_argument("--api-key", default=os.environ.get("YOUTUBE_API_KEY"),
        help="YouTube Data API key (or YOUTUBE_API_KEY in .env)")
    idx.add_argument("--db-path", default="./embedclipfarm_db",
        help="Custom output path (default: auto-derived from source)")
    idx.add_argument("--chunk-seconds", type=int, default=30,
        help="Transcript chunk size in seconds (default: 30)")
    idx.add_argument("--max-videos", type=int, default=200,
        help="Max videos from playlist/channel (default: 200)")
    idx.add_argument("--clip", action="store_true",
        help="CLIP keyframe embeddings (local, no API)")
    idx.add_argument("--frame-interval", type=int, default=30,
        help="Seconds between keyframes (default: 30)")
    idx.add_argument("--max-frames", type=int, default=20,
        help="Max keyframes per video (default: 20)")
    idx.add_argument("--whisper", action="store_true",
        help="Whisper for videos without captions")
    idx.add_argument("--whisper-model", default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)")
    idx.add_argument("--speaker-id", action="store_true",
        help="Speaker diarization (requires pyannote-audio + HF_TOKEN)")
    idx.add_argument("--no-punctuate", action="store_true",
        help="Disable automatic punctuation restoration")
    idx.add_argument("--vision", default=None, choices=["gemini", "claude"],
        help="Annotate keyframes with vision API (descriptions fused with transcripts)")
    idx.add_argument("--vision-key", default=None,
        help="API key for vision (or GEMINI_API_KEY / ANTHROPIC_API_KEY in .env)")
    idx.add_argument("--vision-prompt", default=None,
        help="Custom prompt for vision annotation")
    idx.add_argument("--vision-qa", action="store_true",
        help="Use Q&A-style prompt (objects, text, people, setting, action)")
    idx.add_argument("--cookies-from-browser", default=None,
        help="Browser cookies for age-restricted videos (chrome/firefox/safari)")
    idx.add_argument("--show-transcripts", action="store_true",
        help="Print transcripts during indexing")

    # --- search ---
    srch = sub.add_parser("search", help="Search indexed videos")
    srch.add_argument("query", help="Search query")
    srch.add_argument("--mode", default="text", choices=["text", "visual", "both"],
        help="Search mode (default: text)")
    srch.add_argument("--top-k", type=int, default=10,
        help="Number of results (default: 10)")
    srch.add_argument("--db-path", default="./embedclipfarm_db",
        help="ChromaDB path (auto-detected)")
    srch.add_argument("--download", default=None, metavar="DIR",
        help="Download all matching clips to directory")
    srch.add_argument("--srt", default=None, metavar="FILE",
        help="Export results as SRT subtitle file")
    srch.add_argument("--compile", default=None, metavar="FILE",
        help="Stitch all matching clips into one video")
    srch.add_argument("--filter-video", default=None,
        help="Filter results to a specific video ID")
    srch.add_argument("--filter-type", default=None,
        choices=["text", "fused", "gemini", "claude", "whisper"],
        help="Filter results by source type")
    srch.add_argument("--min-score", type=float, default=0,
        help="Minimum similarity score (default: 0)")
    srch.add_argument("--cookies-from-browser", default=None,
        help="Browser cookies for clip downloads")

    # --- shell ---
    sh = sub.add_parser("shell", help="Interactive search (loads model once)")
    sh.add_argument("--top-k", type=int, default=10,
        help="Results per query (default: 10)")
    sh.add_argument("--db-path", default="./embedclipfarm_db",
        help="ChromaDB path (auto-detected)")

    # --- merge ---
    mg = sub.add_parser("merge", help="Merge multiple project indexes")
    mg.add_argument("sources", nargs="+",
        help="Project directories to merge (e.g. ./channel1 ./channel2)")
    mg.add_argument("--output", required=True,
        help="Output directory for merged index")

    # --- clip ---
    cl = sub.add_parser("clip", help="Download a specific clip from YouTube")
    cl.add_argument("video", help="YouTube video URL or ID")
    cl.add_argument("--start", type=float, required=True, help="Start time in seconds")
    cl.add_argument("--end", type=float, required=True, help="End time in seconds")
    cl.add_argument("--padding", type=float, default=2,
        help="Seconds of padding (default: 2)")
    cl.add_argument("--output-dir", default="./clips",
        help="Output directory (default: ./clips)")
    cl.add_argument("--cookies-from-browser", default=None,
        help="Browser cookies for age-restricted videos")

    # --- export ---
    exp = sub.add_parser("export", help="Export index as JSON for web UI")
    exp.add_argument("--output", default="embedclipfarm_index.json",
        help="Output file")
    exp.add_argument("--db-path", default="./embedclipfarm_db",
        help="ChromaDB path (auto-detected)")

    # --- transcripts ---
    tr = sub.add_parser("transcripts", help="View or save transcripts")
    tr.add_argument("--video", default=None,
        help="Specific video ID (default: all)")
    tr.add_argument("--save", default=None,
        help="Save as .txt files to directory")
    tr.add_argument("--db-path", default="./embedclipfarm_db",
        help="ChromaDB path (auto-detected)")

    # --- serve ---
    sv = sub.add_parser("serve", help="Local server for web UI clip downloads")
    sv.add_argument("--port", type=int, default=8899,
        help="Port (default: 8899)")
    sv.add_argument("--cookies-from-browser", default=None,
        help="Browser cookies for clip downloads")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "index": cmd_index,
        "search": cmd_search_and_clip,
        "shell": cmd_shell,
        "merge": cmd_merge,
        "clip": cmd_clip,
        "export": cmd_export,
        "transcripts": cmd_transcripts,
        "serve": cmd_serve,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
