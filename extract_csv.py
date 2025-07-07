import csv, os, sys, traceback
from pathlib import Path
from mido import MidiFile
from tqdm import tqdm

SOURCE_DIRS = [
    r"Your/Files/Pathway",
    r"Your/Files/Pathway",
]
OUTPUT_DIR = Path(r"Your/Files/Pathway")
LOG_FILE   = OUTPUT_DIR / "midi_export_errors.log"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLS = ["file", "track", "abs_time_ticks", "delta_ticks", "type"]
COMMON_ATTRS = [
    "channel", "note", "velocity", "program",
    "control", "value", "tempo", "numerator",
    "denominator", "key", "text",
]

def all_midis(dirs):
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".mid", ".midi")):
                    yield Path(root) / f

def load_midi(path):

    # clip=True corrige bytes 128‑255 → 127
    return MidiFile(path, clip=True)

def export_csv(path: Path):
    mid = load_midi(path)
    rows, extras = [], set()
    abs_times = [0] * len(mid.tracks)

    for tidx, track in enumerate(mid.tracks):
        for msg in track:
            abs_times[tidx] += msg.time
            row = {
                "file": path.name,
                "track": tidx,
                "abs_time_ticks": abs_times[tidx],
                "delta_ticks": msg.time,
                "type": msg.type,
            }
            for k, v in msg.dict().items():
                if k not in ("time", "type"):
                    row[k] = v
                    extras.add(k)
            rows.append(row)

    cols = BASE_COLS + sorted(extras | set(COMMON_ATTRS))
    for r in rows:
        for c in cols:
            r.setdefault(c, "")

    out = OUTPUT_DIR / (path.stem + ".csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=cols).writeheader()
        csv.DictWriter(fh, fieldnames=cols).writerows(rows)

    return len(mid.tracks), len(rows)

def main():
    midi_paths = list(all_midis(SOURCE_DIRS))
    if not midi_paths:
        print(" No MIDI files found.")
        return

    err_log = LOG_FILE.open("w", encoding="utf-8")
    ok, skipped = 0, 0

    print(f"Processing {len(midi_paths)} MIDI files")
    for p in tqdm(midi_paths, unit="file"):
        try:
            n_tracks, n_events = export_csv(p)
            ok += 1
            tqdm.write(f" {p.name}: {n_tracks} tracks, {n_events} events")
        except Exception as e:
            skipped += 1
            tqdm.write(f" {p.name} skipped  ({e})")
            traceback.print_exc(file=err_log)

    err_log.close()
    print(
        f"\n Finished. Converted: {ok}, Skipped: {skipped}. "
        f"Details in {LOG_FILE if skipped else 'no errors'}"
    )

if __name__ == "__main__":
    main()
