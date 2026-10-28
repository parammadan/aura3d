"""Stitch narration + clips into the walkthrough video.

Each segment: its clip, padded/trimmed to (voice + 1.4 s), voice starting at
0.6 s, short fades at both ends. Segments without a clip or audio are skipped
(and reported), so a draft can be cut before every piece exists.

    python assemble.py [out.mp4]
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

W = Path(__file__).parent
LEAD, TAIL = 0.6, 0.8


def probe(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(p)]).decode())


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else W / "walkthrough.mp4"
    segs = json.loads((W / "script.json").read_text())
    tmp = Path(tempfile.mkdtemp())
    parts, skipped, total = [], [], 0.0
    for s in segs:
        clip, voice = W / "clips" / f"{s['id']}.mp4", W / "audio" / f"{s['id']}.mp3"
        if not clip.exists() or not voice.exists():
            skipped.append(s["id"]); continue
        d = probe(voice) + LEAD + TAIL
        part = tmp / f"{s['id']}.mp4"
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(clip), "-i", str(voice),
            "-filter_complex",
            f"[0:v]fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
            f"tpad=stop_mode=clone:stop_duration=30,trim=duration={d:.3f},setpts=PTS-STARTPTS,"
            f"fade=t=in:st=0:d=0.25,fade=t=out:st={d - 0.25:.3f}:d=0.25,format=yuv420p[v];"
            f"[1:a]adelay={int(LEAD * 1000)}|{int(LEAD * 1000)},apad,atrim=duration={d:.3f},"
            f"aformat=sample_rates=48000:channel_layouts=stereo[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k", str(part)], check=True)
        parts.append(part); total += d
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    print(f"{out}: {len(parts)} segments, {total:.1f}s" + (f" (skipped: {', '.join(skipped)})" if skipped else ""))


if __name__ == "__main__":
    main()
