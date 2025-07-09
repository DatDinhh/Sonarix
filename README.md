# Sonarix
Sonarix is a lightweight Transformer pipeline that turns any folder of MIDI/CSV files into a playable, brand-new score. No DAW macros, no black-box APIs.
## Description
This repository contains a **lean, three‑part workflow** for symbolic‑music generation:

1. **MIDI → CSV extraction** - converts raw Standard MIDI Files into event‑level tables.  
2. **Compact GPT model** - a 25 M‑parameter Transformer (RMSNorm + RoPE + SwiGLU) defined in <300 LOC.  
3. **Inference & MIDI rebuild** - samples new token sequences and writes a valid `.mid`.

The code targets researchers and hobbyists who want an end‑to‑end example without the bloat of full‑scale music‑AI frameworks.

---

## Features
* **Event‑granular CSV** with absolute ticks, delta ticks, and all common MIDI fields.
* **Lightweight architecture** - 6 decoder blocks, 8 heads, no external dependencies beyond PyTorch.
* **Temperature + top‑k sampling** for controllable creativity.
* **Deterministic decoder** that guarantees note‑on/off pairing and adjustable tempo scaling.
* Runs in real‑time on a single GPU (CPU also works for small pieces).

---

## File Structure
| File | Purpose |
|------|---------|
| `extract_csv.py` | Walks a directory tree and dumps one CSV per MIDI file. |
| `model.py` | Defines the GPT‑like Transformer used for inference. |
| `generate.py` | Loads a checkpoint, samples tokens, converts them back to MIDI. |
| `requirements.txt` | Minimal Python dependencies. |
| `README.md` | This document. |

*(A trained checkpoint `best.pt` should be placed in `runs/` before generation.)*

---

## Future Improvements
Full training script with mixed‑precision and gradient‑checkpointing.
Velocity & duration tokens for richer expression.
Multi‑track output and drum‑channel support.
Edge‑case MIDI handling (sysex, NRPN, meta lyrics).

## How to Run (Python ≥ 3.10)
```bash
# 1. Install deps
pip install -r requirements.txt        # torch, mido, pretty_midi, numpy, pandas, tqdm

# 2. Extract MIDI to CSV
python src/extract_csv.py \
  --src  data/midi     \
  --dst  data/csv      \
  --log  extract.log

# 3. Generate new music   (requires runs/best.pt and data/cache.npy)
python src/generate.py \
  --ckpt       runs/best.pt \
  --cache      data/cache.npy \
  --out        outputs/      \
  --seed       NOTE_ON_64 \
  --tokens     1500



