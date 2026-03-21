# EmbedClipFarm

Semantic search over YouTube content — search by meaning, not keywords.

EmbedClipFarm indexes YouTube video transcripts as vector embeddings, letting you search across videos by meaning rather than exact keyword matches. Ask "guy having fun" and find relevant moments even if those exact words are never spoken.

## How It Works

```
YouTube URLs → Transcripts → Chunked Text → Vector Embeddings → Searchable Index
```

### Pipeline

1. **Fetch transcripts** — Uses `yt-dlp` to pull auto-generated captions directly from YouTube (no video download needed). Falls back to `youtube-transcript-api` if available.

2. **Chunk transcripts** — Splits transcripts into ~30-second segments so search results point to specific moments in a video, not just "somewhere in this 2-hour podcast."

3. **Embed chunks** — Each text chunk is converted into a 384-dimensional vector using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). These vectors capture the *meaning* of the text, not just the words.

4. **Store & export** — Embeddings are stored in ChromaDB and auto-exported as `index.json`. Transcripts are saved as readable `.txt` files.

5. **Search** — From the terminal or the web UI. Results are ranked by semantic similarity and link directly to the YouTube timestamp.

### Optional Features

- **CLIP visual search** — Sample keyframes from videos and embed them with [CLIP](https://github.com/mlfoundations/open_clip). Search by visual description.
- **Whisper fallback** — For videos without captions, download audio and transcribe with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- **Browser cookies** — Access age-restricted videos using your browser's YouTube session.

## Quick Start

### Install

```bash
pip install youtube-transcript-api chromadb sentence-transformers numpy python-dotenv yt-dlp
```

### Index a channel

```bash
python embedclipfarm.py index "@omalleyrock" --max-videos 10
```

This creates an output folder with everything:

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

  [0.242] Pipe rock theory
         Video: https://youtube.com/watch?v=yzuj14hzCaw&t=199
         Time:  3:19 → 3:51
         Text:  Look, they're letting caterers in. We'll get you in as a caterer...

  [0.207] Las Vegas show 3/19
         Video: https://youtube.com/watch?v=MVC8WQSutyQ&t=25
         Time:  0:25 → 0:39
         Text:  incredible handsfree orgasm wow yeah will you survive...
```

Each result includes a direct YouTube link with timestamp — click it to jump to that exact moment.

### Search from the web UI

1. Open [**artificialnouveau.com/miniprojects/embedclipfarm**](https://www.artificialnouveau.com/miniprojects/embedclipfarm/) (or `index.html` locally)
2. Drop your `index.json` onto the page (e.g. `./omalleyrock/index.json`)
3. Type a query — results show video thumbnails, transcript excerpts, and scores
4. Click any result to jump to that moment on YouTube

## Index Multiple Channels

Each source gets its own subfolder automatically:

```bash
python embedclipfarm.py index "@omalleyrock" --max-videos 10
python embedclipfarm.py index "@channelhandle2"
python embedclipfarm.py index "https://youtube.com/playlist?list=PLxxxxx"
```

Result:

```
./omalleyrock/
  ├── index.json
  ├── transcripts/
  └── db/
./channelhandle2/
  ├── index.json
  ├── transcripts/
  └── db/
./PLxxxxx/
  ├── index.json
  ├── transcripts/
  └── db/
```

Load any `index.json` into the web UI to search that channel. With one database, `search` auto-detects it. With multiple, specify which:

```bash
# Auto-detects if only one exists
python embedclipfarm.py search "topic"

# Specify which database
python embedclipfarm.py search "topic" --db-path ./omalleyrock/db
```

## More Examples

```bash
# Show transcripts as they're indexed
python embedclipfarm.py index "@omalleyrock" --show-transcripts

# With Whisper for videos without captions
python embedclipfarm.py index "@omalleyrock" --whisper

# With browser cookies for age-restricted videos
python embedclipfarm.py index "@omalleyrock" --cookies-from-browser chrome

# Everything at once
python embedclipfarm.py index "@omalleyrock" --max-videos 20 --whisper --cookies-from-browser chrome --show-transcripts

# Index individual videos
python embedclipfarm.py index "https://youtube.com/watch?v=VIDEO_ID"

# Multiple videos merged into one index
python embedclipfarm.py index "URL1" "URL2" "URL3"

# From a file of URLs
python embedclipfarm.py index urls.txt
```

### View transcripts

```bash
# Print all transcripts
python embedclipfarm.py transcripts

# Specific video
python embedclipfarm.py transcripts --video yzuj14hzCaw

# Save to files (also auto-done during indexing)
python embedclipfarm.py transcripts --save ./output
```

### API key (for playlists/channels)

Indexing individual videos works without an API key. For playlists or channels, create a free [YouTube Data API key](https://console.cloud.google.com/apis/credentials):

```bash
echo "YOUTUBE_API_KEY=your_key_here" > .env
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
  --db-path PATH ChromaDB path (auto-detected if only one exists)
```

### `transcripts`

```
python embedclipfarm.py transcripts [options]

Options:
  --video ID     Show transcript for a specific video ID (default: all)
  --save DIR     Save transcripts as .txt files
  --db-path PATH ChromaDB path (auto-detected if only one exists)
```

### `export`

```
python embedclipfarm.py export [options]

Options:
  --output FILE  Output JSON file (default: embedclipfarm_index.json)
  --db-path PATH ChromaDB path (auto-detected if only one exists)
```

## Architecture

```
embedclipfarm.py     — Single-file CLI (Python)
index.html           — Web search UI (load index.json → search → results)
requirements.txt     — Python dependencies
.env                 — YouTube API key (not committed)
```

### Data flow

```
                            CLI
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  YouTube     │───→│  yt-dlp      │───→│  Sentence    │───→│ project/ │
│  @channel    │    │  transcripts │    │  Transformer │    │  db/     │
└─────────────┘    └──────────────┘    │  embeddings  │    │  index.json
                                       └──────────────┘    │  transcripts/
                                                           └──────────┘
                           Web UI                               │
┌─────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  Query       │───→│  Load        │───→│  Cosine      │←──────┘
│  "topic X"   │    │  index.json  │    │  similarity  │───→ Ranked results
└─────────────┘    └──────────────┘    └──────────────┘    with YT timestamps
```

## License

MIT
