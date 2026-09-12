"""Download sample videos to test against. Run with:  python get_sample_videos.py

Videos are not committed. They are large, they are not mine to redistribute,
and .gitignore excludes them, so this script is how a fresh clone gets
something to run on. Files already present are left alone.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

VIDEO_DIR = Path(__file__).parent / "videos"

SAMPLES = {
    "people-detection.mp4": (
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
        "People walking through an indoor space. The default test clip.",
    ),
    "person-bicycle-car-detection.mp4": (
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
        "person-bicycle-car-detection.mp4",
        "A street scene with three different classes moving at once.",
    ),
}


_last_pct = -1


def report(done: int, block: int, total: int) -> None:
    """Print progress only when the whole number percentage changes.

    urlretrieve calls this once per block, which is hundreds of times for a
    5 MB file. Printing every call floods a log with identical lines whenever
    output is captured rather than shown on a terminal.
    """
    global _last_pct
    if total <= 0:
        return
    pct = min(100, done * block * 100 // total)
    if pct != _last_pct:
        _last_pct = pct
        print(f"\r  {pct:3d}%", end="", flush=True)


def main() -> int:
    VIDEO_DIR.mkdir(exist_ok=True)
    failed = 0

    for name, (url, description) in SAMPLES.items():
        target = VIDEO_DIR / name
        if target.exists():
            print(f"{name}: already present ({target.stat().st_size / 1e6:.1f} MB)")
            continue

        print(f"{name}: {description}")
        try:
            # Download to a temporary name and rename only on success, so an
            # interrupted download cannot leave a truncated file that looks
            # complete on the next run.
            partial = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(url, partial, reporthook=report)
            partial.replace(target)
            print(f"\r  saved {target.stat().st_size / 1e6:.1f} MB to {target}")
        except Exception as exc:
            failed += 1
            print(f"\r  failed: {exc}")

    if failed:
        print(f"\n{failed} download(s) failed. You can also point --source at any video file.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
