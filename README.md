# semtune

Turns text into a piano score.

Give it any English or German text. It produces a LilyPond score, a printable PDF, a MIDI file, and a per-note log that explains every choice the system made.

The core idea is simple: words that mean similar things become music that sounds similar. When a word echoes one from earlier in the text, the system replays the earlier note material — transformed in a way that depends on whether the emotional context has stayed the same or shifted.

## Requirements

- Python 3.10 or newer
- [LilyPond](https://lilypond.org) on PATH (`brew install lilypond` on macOS)

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Optional but recommended: fetch the NRC-VAD emotion lexicons (one-time, around 5 MB). With them, dynamics and register actually track the words' valence and arousal.

```bash
python scripts/download_lexicons.py
```

## Run

```bash
python -m semtune --text resources/texts/04_stein_sacred_emily.txt --seed 42 --out output/sacred_emily/
```

You get five files under `--out`:

| file | what it is |
|---|---|
| `score.ly` | LilyPond source |
| `score.pdf` | Engraved score |
| `score.midi` | MIDI file |
| `trace.md` | Per-note log: which rule fired and why |
| `config.json` | All settings used for this run |

Same text plus same seed always produces the same output, down to the byte.

## Trying it on your own text

```bash
echo "Your text goes here." > my_text.txt
python -m semtune --text my_text.txt --seed 42 --out output/mine/
```

Open `output/mine/score.pdf` to see the score, and `output/mine/trace.md` to read why each note was chosen.

Three sample texts ship in `resources/texts/`:

- `01_ball_karawane.txt` — Hugo Ball's Dada sound-poetry (shows the phonetic fallback)
- `02_fontane_ueberlass_es_der_zeit.txt` — Fontane verse in German
- `04_stein_sacred_emily.txt` — Stein's *Rose is a rose…* (shows the recurrence rule)

## Tuning

Every knob lives in `src/semtune/config.py`. The ones you'll actually want to touch:

| knob | what changes |
|---|---|
| `seed` | Which random branch is taken at each step |
| `theta_quote` | How similar two words need to be before the system "echoes" the earlier one. Lower means more echoes. |
| `kappa` | How literally those echoes are reproduced. Higher means closer to the original. |
| `lambda_warp` | How strongly similar-meaning words land on close pitches |
| `tempo_bpm` | Score tempo |

The CLI exposes `--seed`, `--tempo`, and `--theta-quote` directly. Everything else needs editing `config.py`.

## How it works

Three steps from text to score:

1. **Read the text.** Split it into words, embed each word with a multilingual sentence-transformer, group related words into clusters, and look up each word's emotional rating (valence, arousal, dominance). Compute how similar every word is to every other word; this similarity table is what drives recurrence later.

2. **Choose the notes.** A Markov chain picks a pitch class and a duration for each word. The choice is biased by:
   - which cluster the word belongs to
   - how similar this word is to the previous one (close meaning leans toward small intervals; distant meaning leans toward larger leaps)
   - the word's arousal (excited words lean toward shorter notes)
   - whether this word echoes an earlier word — if so, the chain is pushed toward replaying the earlier note material

3. **Write the score.** Translate the notes into LilyPond, render the PDF and MIDI, and log every decision in `trace.md`.

Full detail in [`docs/system_documentation.md`](docs/system_documentation.md).

## Repository layout

```
src/semtune/      source package
docs/             documentation
resources/texts/  sample texts
scripts/          lexicon downloader and batch runner
```

## Author

Jayasankar Kumar Santhirani · <jayasankarkumars@gmail.com>
