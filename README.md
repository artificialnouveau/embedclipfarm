# EmbedClipFarm

Semantic search over YouTube content — search by meaning, not keywords.

EmbedClipFarm indexes YouTube video transcripts as vector embeddings, letting you search across videos by meaning rather than exact keyword matches. Ask "guy having fun" and find relevant moments even if those exact words are never spoken.

## How It Works

```
YouTube URLs → Transcripts + Keyframes → Embeddings → Searchable Index
```

### What gets analyzed

| Source | What it does | Flag |
|--------|-------------|------|
| **Transcripts** | Auto-captions from YouTube via yt-dlp | *(default)* |
| **Whisper** | Local speech-to-text for videos without captions | `--whisper` |
| **Gemini vision** | Sends keyframes to Gemini API for rich scene descriptions | `--vision gemini` |
| **Claude vision** | Sends keyframes to Claude API for rich scene descriptions | `--vision claude` |
| **CLIP** | Local keyframe embeddings for visual search (no API needed) | `--clip` |
| **Speaker ID** | Tags transcripts with speaker labels | `--speaker-id` |

### How vision annotation works

When you use `--vision gemini` or `--vision claude`, the CLI:
1. Extracts keyframes from each video (1 frame every 30s by default)
2. Sends each frame to the vision API, which returns a description like *"A man in a parking lot gesturing animatedly at the camera, wearing a gray hoodie. Behind him is a strip mall with neon signs."*
3. Those descriptions are embedded as **regular text chunks** alongside transcript chunks
4. So you can search by what was **said** and what was **shown** in a single query
5. Each result is tagged with its source (transcript / gemini / claude / whisper) so you know where it came from

This is different from `--clip`, which embeds frames directly into a separate visual collection searched with `--mode visual`. For the web UI, `--vision` is the better choice since everything ends up in one searchable index.

### Pipeline

1. **Fetch transcripts** — Uses `yt-dlp` to pull auto-generated captions directly from YouTube (no video download needed). Falls back to `youtube-transcript-api`.

2. **Chunk transcripts** — Splits into ~30-second segments so results point to specific moments.

3. **Embed chunks** — Each chunk is converted to a 384-dimensional vector using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) that captures its semantic meaning.

4. **Store & export** — Stored in ChromaDB and auto-exported as `index.json` with transcripts saved as `.txt` files.

5. **Search** — Your query is embedded the same way and ranked by cosine similarity. Results link directly to the YouTube timestamp.

## Quick Start

### Install

```bash
pip install youtube-transcript-api chromadb sentence-transformers numpy python-dotenv yt-dlp
```

### API keys (in `.env` file)

```bash
# Required for playlists/channels
YOUTUBE_API_KEY=your_key_here

# Optional — for vision scene annotation (pick one)
GEMINI_API_KEY=your_key_here      # free at https://aistudio.google.com/apikey
ANTHROPIC_API_KEY=your_key_here   # from https://console.anthropic.com

# Optional — for speaker diarization
HF_TOKEN=your_token_here          # from https://huggingface.co/settings/tokens
```

### Index a channel

```bash
python embedclipfarm.py index "@omalleyrock" --max-videos 10
```

Output:

```
./omalleyrock/
  ├── index.json       ← load this in the web UI to search
  ├── transcripts/     ← readable .txt files for every video
  │   ├── Slugs_wYrNjPGgAAA.txt
  │   ├── Pipe_rock_theory_yzuj14hzCaw.txt
  │   └── ...
  └── db/              ← ChromaDB vector database (for CLI search)
```

### Search from the terminal

```bash
python embedclipfarm.py search "guy having fun"
```

```
Auto-detected database: omalleyrock/db

============================================================
TEXT SEARCH: "guy having fun"
============================================================

  [1] [0.242] Pipe rock theory
         Video: https://youtube.com/watch?v=yzuj14hzCaw&t=199
         Time:  3:19 → 3:51
         Text:  Look, they're letting caterers in. We'll get you in as a caterer...
         Clip:  python embedclipfarm.py clip yzuj14hzCaw --start 199 --end 231
```

### Search from the web UI

1. Open [**artificialnouveau.com/miniprojects/embedclipfarm**](https://www.artificialnouveau.com/miniprojects/embedclipfarm/) (or `index.html` locally)
2. Drop your `index.json` onto the page
3. Type a query — results show embedded YouTube player at the matching timestamp, transcript excerpt, and relevance score
4. Click **Copy clip cmd** to download that exact segment

### Download clips

```bash
# Download a specific clip
python embedclipfarm.py clip yzuj14hzCaw --start 199 --end 231

# Search and auto-download all matching clips
python embedclipfarm.py search "slugs" --download ./clips
```

## Index Multiple Channels

Each source gets its own subfolder automatically:

```bash
python embedclipfarm.py index "@omalleyrock" --max-videos 10
python embedclipfarm.py index "@channelhandle2"
python embedclipfarm.py index "https://youtube.com/playlist?list=PLxxxxx"
```

```
./omalleyrock/        → index.json + transcripts/ + db/
./channelhandle2/     → index.json + transcripts/ + db/
./PLxxxxx/            → index.json + transcripts/ + db/
```

Load any `index.json` into the web UI. With one database, `search` auto-detects it. With multiple, specify which:

```bash
python embedclipfarm.py search "topic" --db-path ./omalleyrock/db
```

## All Indexing Options

```bash
# Single video
python embedclipfarm.py index "https://youtube.com/watch?v=VIDEO_ID"

# Multiple videos
python embedclipfarm.py index "URL1" "URL2" "URL3"

# Channel
python embedclipfarm.py index "@channelname" --max-videos 20

# Playlist
python embedclipfarm.py index "https://youtube.com/playlist?list=PLxxxxx"

# From a file (txt, csv, or json)
python embedclipfarm.py index urls.txt
python embedclipfarm.py index videos.csv
python embedclipfarm.py index playlist.json

# Show transcripts during indexing
python embedclipfarm.py index "@channel" --show-transcripts

# With Whisper for videos without captions
python embedclipfarm.py index "@channel" --whisper

# With speaker identification (requires pyannote-audio + HF_TOKEN)
python embedclipfarm.py index "@channel" --whisper --speaker-id

# With Gemini scene annotation
python embedclipfarm.py index "@channel" --vision gemini

# With Claude scene annotation
python embedclipfarm.py index "@channel" --vision claude

# With a custom vision prompt (focus on specific aspects)
python embedclipfarm.py index "@channel" --vision gemini --vision-prompt "Describe the fashion and clothing visible in this frame. Note colors, styles, brands, and accessories."

# More custom prompt examples
python embedclipfarm.py index "@channel" --vision claude --vision-prompt "List all text visible on screen. Describe any graphics, logos, or data visualizations."
python embedclipfarm.py index "@channel" --vision gemini --vision-prompt "Describe the emotions, body language, and facial expressions of people in this frame."

# With CLIP visual embeddings (local, no API)
python embedclipfarm.py index "@channel" --clip

# With browser cookies for age-restricted videos
python embedclipfarm.py index "@channel" --cookies-from-browser chrome

# Disable automatic punctuation
python embedclipfarm.py index "@channel" --no-punctuate

# Everything at once
python embedclipfarm.py index "@channel" --max-videos 20 --whisper --speaker-id --vision gemini --cookies-from-browser chrome --show-transcripts
```

### File input formats

**Text file** (`urls.txt`) — one URL or video ID per line:
```
https://youtube.com/watch?v=VIDEO_ID1
https://youtube.com/watch?v=VIDEO_ID2
dQw4w9WgXcQ
```

**CSV** (`videos.csv`) — extracts YouTube URLs/IDs from any column:
```
title,url
"Video 1","https://youtube.com/watch?v=VIDEO_ID1"
"Video 2","https://youtube.com/watch?v=VIDEO_ID2"
```

**JSON** (`playlist.json`) — list of URLs or objects:
```json
["https://youtube.com/watch?v=ID1", "https://youtube.com/watch?v=ID2"]
```
or:
```json
{"videos": [{"url": "https://youtube.com/watch?v=ID1"}, {"url": "https://youtube.com/watch?v=ID2"}]}
```

## View Transcripts

```bash
# Print all transcripts (also auto-saved during indexing)
python embedclipfarm.py transcripts

# Specific video
python embedclipfarm.py transcripts --video yzuj14hzCaw

# Save to files
python embedclipfarm.py transcripts --save ./output
```

## CLI Reference

### `index`

```
python embedclipfarm.py index <source> [source2 ...] [options]

Options:
  --api-key KEY              YouTube Data API key (or YOUTUBE_API_KEY in .env)
  --db-path PATH             Custom output path (default: auto-derived from source)
  --chunk-seconds N          Transcript chunk size in seconds (default: 30)
  --max-videos N             Max videos from playlist/channel (default: 200)
  --show-transcripts         Print transcripts during indexing
  --cookies-from-browser B   Browser for age-restricted videos (chrome/firefox/safari)
  --vision {gemini,claude}   Annotate keyframes with vision API
  --vision-key KEY           API key for vision (or set in .env)
  --vision-prompt TEXT       Custom prompt for scene annotation (see below)
  --clip                     CLIP keyframe embeddings (local, no API)
  --frame-interval N         Seconds between keyframes (default: 30)
  --max-frames N             Max keyframes per video (default: 20)
  --whisper                  Whisper for videos without captions
  --whisper-model SIZE       tiny/base/small/medium/large-v3 (default: base)
  --speaker-id               Speaker diarization (requires pyannote-audio + HF_TOKEN)
  --no-punctuate             Disable automatic punctuation (on by default)
```

### `search`

```
python embedclipfarm.py search <query> [options]

Options:
  --mode {text,visual,both}  Search mode (default: text)
  --top-k N                  Number of results (default: 10)
  --download DIR             Download all matching clips to directory
  --cookies-from-browser B   Browser cookies for clip downloads
  --db-path PATH             ChromaDB path (auto-detected if one exists)
```

### `clip`

```
python embedclipfarm.py clip <video> --start N --end N [options]

Options:
  --padding N                Seconds of padding (default: 2)
  --output-dir DIR           Output directory (default: ./clips)
  --cookies-from-browser B   Browser cookies for age-restricted videos
```

### `transcripts`

```
python embedclipfarm.py transcripts [options]

Options:
  --video ID     Specific video ID (default: all)
  --save DIR     Save as .txt files
  --db-path PATH ChromaDB path (auto-detected)
```

### `export`

```
python embedclipfarm.py export [options]

Options:
  --output FILE  Output file (default: embedclipfarm_index.json)
  --db-path PATH ChromaDB path (auto-detected)
```

## Architecture

```
embedclipfarm.py     — Single-file CLI (Python)
index.html           — Web search UI (load index.json → search → results)
requirements.txt     — Python dependencies
.env                 — API keys (not committed)
```

### Data flow

```
                            CLI
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  YouTube     │───→│  yt-dlp      │───→│  Sentence    │───→│ project/ │
│  @channel    │    │  transcripts │    │  Transformer │    │  db/     │
└─────────────┘    └──────────────┘    │  embeddings  │    │  index.json
                   ┌──────────────┐    └──────────────┘    │  transcripts/
                   │  Gemini /    │───→  text descriptions  └──────────┘
                   │  Claude /    │     (embedded as text)       │
                   │  CLIP        │                              │
                   └──────────────┘                              │
                           Web UI                                │
┌─────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  Query       │───→│  Load        │───→│  Cosine      │←───────┘
│  "topic X"   │    │  index.json  │    │  similarity  │───→ Ranked results
└─────────────┘    └──────────────┘    └──────────────┘    with YT timestamps
```

## License

MIT
