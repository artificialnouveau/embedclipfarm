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

4. **Store & export** — Embeddings are stored in ChromaDB and auto-exported as `index.json` for the web UI.

5. **Search** — Your query is embedded with the same model, then compared against all stored chunks. Results are ranked by semantic similarity and link directly to the timestamp on YouTube.

### Optional Features

- **CLIP visual search** — Sample keyframes from videos and embed them with [CLIP](https://github.com/mlfoundations/open_clip). Search by visual description ("person at a podium") to find matching frames.
- **Whisper fallback** — For videos without captions, download the audio and transcribe with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- **Browser cookies** — Access age-restricted videos by using your browser's YouTube login session.

## Quick Start

### Install

```bash
pip install youtube-transcript-api chromadb sentence-transformers numpy python-dotenv yt-dlp
```

### Set up API key (for playlists/channels)

Indexing individual videos works without an API key. For playlists or channels, you need a [YouTube Data API key](https://console.cloud.google.com/apis/credentials) (free).

```bash
echo "YOUTUBE_API_KEY=your_key_here" > .env
```

### Index a channel

```bash
python embedclipfarm.py index "@channelhandle"
```

This creates an output folder with everything you need:

```
./channelhandle/
  ├── index.json       ← load this in the web UI to search
  ├── transcripts/     ← readable .txt files for every video
  │   ├── Video_Title_1_VIDEO_ID.txt
  │   └── Video_Title_2_VIDEO_ID.txt
  └── db/              ← ChromaDB vector database (for CLI search)
```

### Search from the terminal

```bash
python embedclipfarm.py search "your query" --db-path ./channelhandle/db
```

### Search from the web UI

1. Open [the web UI](https://www.artificialnouveau.com/miniprojects/embedclipfarm/) (or `index.html` locally)
2. Drop your `index.json` file onto the page
3. Type a query and search — results link directly to the matching YouTube timestamp

## Index Multiple Channels/Videos

Each source gets its own subfolder automatically:

```bash
# Index different channels — each gets its own folder
python embedclipfarm.py index "@channel1"
python embedclipfarm.py index "@channel2"
python embedclipfarm.py index "@channel3"

# Index individual videos
python embedclipfarm.py index "https://youtube.com/watch?v=VIDEO_ID"

# Index a playlist
python embedclipfarm.py index "https://youtube.com/playlist?list=PLxxxxx"
```

Result:

```
./channel1/
  ├── index.json
  ├── transcripts/
  └── db/
./channel2/
  ├── index.json
  ├── transcripts/
  └── db/
./channel3/
  ├── index.json
  ├── transcripts/
  └── db/
./VIDEO_ID/
  ├── index.json
  ├── transcripts/
  └── db/
```

Load any `index.json` into the web UI to search that specific channel/video.

## More Examples

```bash
# Show transcripts as they're indexed
python embedclipfarm.py index "@channelhandle" --show-transcripts

# Limit number of videos
python embedclipfarm.py index "@channelhandle" --max-videos 20

# With Whisper for videos without captions
python embedclipfarm.py index "@channelhandle" --whisper

# With browser cookies for age-restricted videos
python embedclipfarm.py index "@channelhandle" --cookies-from-browser chrome

# With CLIP keyframe visual search
python embedclipfarm.py index "VIDEO_URL" --clip

# Kitchen sink
python embedclipfarm.py index "@channelhandle" --max-videos 20 --whisper --cookies-from-browser chrome --show-transcripts

# From a file of URLs
python embedclipfarm.py index urls.txt

# Multiple sources at once (merged into one index)
python embedclipfarm.py index "VIDEO_URL1" "VIDEO_URL2" "VIDEO_URL3"
```

### View transcripts after indexing

```bash
# Print all indexed transcripts
python embedclipfarm.py transcripts --db-path ./channelhandle/db

# Print transcript for a specific video
python embedclipfarm.py transcripts --video VIDEO_ID --db-path ./channelhandle/db

# Save all transcripts as .txt files (also auto-done during indexing)
python embedclipfarm.py transcripts --save ./output --db-path ./channelhandle/db
```

### Search

```bash
# Text search
python embedclipfarm.py search "economic impact of automation" --db-path ./channelhandle/db

# Visual search (requires --clip during indexing)
python embedclipfarm.py search "person giving speech at podium" --mode visual --db-path ./channelhandle/db

# Both text and visual
python embedclipfarm.py search "climate policy" --mode both --top-k 20 --db-path ./channelhandle/db
```

## CLI Reference

### `index`

```
python embedclipfarm.py index <source> [source2 ...] [options]

Sources:
  YouTube URL (video, playlist, or channel)
  @channelhandle
  File containing URLs (one per line)

Options:
  --api-key KEY              YouTube Data API key (or set YOUTUBE_API_KEY in .env)
  --db-path PATH             Custom output path (default: auto-derived from source)
  --chunk-seconds N          Transcript chunk size in seconds (default: 30)
  --max-videos N             Max videos from playlist/channel (default: 200)
  --show-transcripts         Print transcripts to terminal during indexing
  --cookies-from-browser B   Browser to extract cookies from (chrome, firefox, safari)
  --clip                     Enable CLIP keyframe embedding
  --frame-interval N         Seconds between keyframes (default: 30)
  --max-frames N             Max keyframes per video (default: 20)
  --whisper                  Use Whisper for videos without captions
  --whisper-model SIZE       tiny/base/small/medium/large-v3 (default: base)
```

### `search`

```
python embedclipfarm.py search <query> [options]

Options:
  --mode MODE    text/visual/both (default: text)
  --top-k N      Number of results (default: 10)
  --db-path PATH ChromaDB storage path
```

### `transcripts`

```
python embedclipfarm.py transcripts [options]

Options:
  --video ID     Show transcript for a specific video ID (default: all)
  --save DIR     Save transcripts as .txt files to a directory
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
index.html           — Web search UI (load index.json, search, get results)
requirements.txt     — Python dependencies
.env                 — YouTube API key (not committed)
```

### Data flow

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  YouTube     │───→│  yt-dlp      │───→│  Sentence    │───→│ ChromaDB │
│  URLs        │    │  transcripts │    │  Transformer │    │  + JSON  │
└─────────────┘    └──────────────┘    │  embeddings  │    └──────────┘
                                       └──────────────┘         │
                   ┌──────────────┐                             │
                   │  index.json  │←────────────────────────────┘
                   └──────────────┘
                         │
┌─────────────┐    ┌─────┴────────┐    ┌──────────────┐
│  Query       │───→│  Web UI      │───→│  Ranked      │───→ Click to jump
│  "topic X"   │    │  (browser)   │    │  results     │    to YouTube timestamp
└─────────────┘    └──────────────┘    └──────────────┘
```

## License

MIT
