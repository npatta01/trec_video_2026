# TRECVID 2026 approach film

The film follows the Q1835 weather example from model inputs to generated
answers and multiple-choice rankings, then compares the official test results.
The [paper](../../paper/trecvid2026.pdf) documents the output wording, scoring
key, and metric-specific leaderboard ranks.

- [Story and caption text](story.json)
- [Artwork and screen layout](film.html)
- [Animation timeline](timeline.js)

The clip depiction reuses the workshop’s original illustration. Narration uses
the Kokoro `af_heart` synthetic voice, rendered locally. The film has no test
footage or audio.

## Render

Use Python with `kokoro-onnx==0.6.1` and `soundfile==0.13.1`, Node with
Playwright and Chromium, and FFmpeg with libx264. Place
[`kokoro-v1.0.int8.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx)
and [`voices-v1.0.bin`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin)
in the build directory. Playwright can be installed there or located with
`TRECVID_FILM_PLAYWRIGHT`. Set `TRECVID_FILM_CHROME` to use an installed Chrome
binary.

From the repository root, with `film_build_dir` pointing to that directory:

```bash
python docs/workshop/approach-film/narrate.py "$film_build_dir"
node docs/workshop/approach-film/render.cjs "$film_build_dir" --preview
node docs/workshop/approach-film/render.cjs "$film_build_dir"
```

The render writes `approach.mp4` and `captions.vtt` to the build directory.
The landing page uses the corresponding MP4, captions, and first-scene poster
in [assets](../../../assets/).
