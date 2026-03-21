# EmbedClipFarm

Semantic search over YouTube content — search by meaning, not keywords.

EmbedClipFarm indexes YouTube video transcripts as vector embeddings, letting you search across videos by meaning rather than exact keyword matches. Ask "economic anxiety in rural communities" and find relevant moments even if those exact words are never spoken.

## How It Works

```
YouTube URLs → Transcripts → Chunked Text → Vector Embeddings → Searchable Index
```

### Pipeline

1. **Fetch transcripts** — Uses `yt-dlp` to pull auto-generated captions directly from YouTube (no video download needed). Falls back to `youtube-transcript-api` if available.

2. **Chunk transcripts** — Splits transcripts into ~30-second segments so search results point to specific moments in a video, not just "somewhere in this 2-hour podcast."

3. **Embed chunks** — Each text chunk is converted into a 384-dimensional vector using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). These vectors capture the *meaning* of the text, not just the words.

4. **Store in ChromaDB** — Embeddings are stored in a local vector database for fast cosine similarity search.

5. **Search** — Your query is embedded with the same model, then compared against all stored chunks. Results are ranked by semantic similarity and link directly to the timestamp on YouTube.

### Optional Features

- **CLIP visual search** — Sample keyframes from videos and embed them with [CLIP](https://github.com/mlfoundations/open_clip). Search by visual description ("person at a podium") to find matching frames.
- **Whisper fallback** — For videos without captions, download the audio and transcribe with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Quick Start

### Install

```bash
pip install youtube-transcript-api chromadb sentence-transformers numpy
```

You also need [yt-dlp](https://github.com/yt-dlp/yt-dlp) installed:

```bash
pip install yt-dlp
# or: brew install yt-dlp
```

### Set up API key (for playlists/channels)

Indexing individual videos works without an API key. For playlists or channels, you need a [YouTube Data API key](https://console.cloud.google.com/apis/credentials) (free).

Create a `.env` file in the project directory:

```bash
echo "YOUTUBE_API_KEY=your_key_here" > .env
```

That's it — the CLI reads it automatically. You can also pass `--api-key` or set the env var directly.

### Index videos

```bash
# Single video
python embedclipfarm.py index "https://www.youtube.com/watch?v=VIDEO_ID"

# Multiple videos
python embedclipfarm.py index "https://youtube.com/watch?v=ID1" "https://youtube.com/watch?v=ID2"

# Entire playlist (requires YouTube Data API key — see below)
python embedclipfarm.py index "https://youtube.com/playlist?list=PLxxxxx"

# Channel
python embedclipfarm.py index "@channelhandle"

# From a file of URLs
python embedclipfarm.py index urls.txt

# With Whisper for videos without captions
python embedclipfarm.py index "VIDEO_URL" --whisper

# With CLIP keyframe visual search
python embedclipfarm.py index "VIDEO_URL" --clip
```

### Search

```bash
# Text search (default)
python embedclipfarm.py search "economic impact of automation"

# Visual search (requires --clip during indexing)
python embedclipfarm.py search "person giving speech at podium" --mode visual

# Both text and visual
python embedclipfarm.py search "climate policy" --mode both --top-k 20
```

### Export for web UI

```bash
python embedclipfarm.py export --output index.json
```

Then open `index.html` in a browser and load the JSON file to search with a visual interface.

## Web UI

The included `index.html` is a standalone web page that provides:

- **Import/Export** — Load a JSON index exported from the CLI
- **In-browser search** — Uses [Transformers.js](https://huggingface.co/docs/transformers.js) to embed your query with the same model (all-MiniLM-L6-v2) and compute similarity client-side
- **Scene Detection** — Upload a local video file for visual scene detection with optional Whisper transcription, all running in-browser
- **Results** — Each result links to the exact timestamp on YouTube

### Live demo

[artificialnouveau.com/miniprojects/embedclipfarm](https://www.artificialnouveau.com/miniprojects/embedclipfarm/)

## CLI Reference

### `index`

```
python embedclipfarm.py index <source> [source2 ...] [options]

Sources:
  YouTube URL (video, playlist, or channel)
  @channelhandle
  File containing URLs (one per line)

Options:
  --api-key KEY          YouTube Data API key (needed for playlists/channels)
  --db-path PATH         ChromaDB storage path (default: ./embedclipfarm_db)
  --chunk-seconds N      Transcript chunk size in seconds (default: 30)
  --max-videos N         Max videos from playlist/channel (default: 200)
  --clip                 Enable CLIP keyframe embedding
  --frame-interval N     Seconds between keyframes (default: 30)
  --max-frames N         Max keyframes per video (default: 20)
  --whisper              Use Whisper for videos without captions
  --whisper-model SIZE   tiny/base/small/medium/large-v3 (default: base)
```

### `search`

```
python embedclipfarm.py search <query> [options]

Options:
  --mode MODE    text/visual/both (default: text)
  --top-k N      Number of results (default: 10)
  --db-path PATH ChromaDB storage path
```

### `export`

```
python embedclipfarm.py export [options]

Options:
  --output FILE  Output JSON file (default: embedclipfarm_index.json)
  --db-path PATH ChromaDB storage path
```

## Architecture

```
embedclipfarm.py     — Single-file CLI (Python)
index.html           — Web search UI (standalone, no build step)
worker.js            — Optional Cloudflare Worker proxy for web auto-fetch
requirements.txt     — Python dependencies
```

### Data flow

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  YouTube     │───→│  yt-dlp      │───→│  Sentence    │───→│ ChromaDB │
│  URLs        │    │  transcripts │    │  Transformer │    │  vectors │
└─────────────┘    └──────────────┘    │  embeddings  │    └──────────┘
                                       └──────────────┘         │
┌─────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  Query       │───→│  Same model  │───→│  Cosine      │←───────┘
│  "topic X"   │    │  embedding   │    │  similarity  │───→ Ranked results
└─────────────┘    └──────────────┘    └──────────────┘    with timestamps
```

## License

MIT
