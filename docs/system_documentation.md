# semtune — System Documentation

A monophonic generator that turns text into a piano score. Same text and same seed always produces the same output, down to the byte.

## What you get

For any English or German text, semtune emits:

- `score.ly` — LilyPond source
- `score.pdf` — engraved score
- `score.midi` — MIDI file
- `trace.md` — a markdown log explaining every note (which rule fired, which cluster the word belongs to, which features applied, and why)
- `config.json` — the exact settings used for this run

Reading `trace.md` is enough to follow the whole run. Every note has a row; every row names the rule responsible for it.

## Pipeline

```
text → semantic → features → decision → events → score → .ly / .pdf / .midi
                                                   └────── trace.md
```

### Semantic layer

Reads the text and turns each word into a feature bundle.

- **Tokenisation.** Splits the text on whitespace and punctuation, preserves case and word order, and records what punctuation follows each word. The punctuation is what places rests later: comma becomes an 8th rest, period a quarter rest, paragraph break a half rest.
- **Language detection.** Picks the English or German emotion lexicon for the downstream lookup.
- **Embedding.** Each word goes through a multilingual sentence-transformer (paraphrase-multilingual-MiniLM-L12-v2) and comes back as a 384-dimensional vector. Words are lowercased at this step so "Rose" and "rose" share a vector; the original case is kept separately for ALL-CAPS detection.
- **Clustering.** KMeans groups the words into a handful of clusters, roughly 2 to 8 depending on text length. Each cluster gets its own Markov transition matrix downstream, so different concept-groups produce different musical material.
- **Emotion lookup.** NRC-VAD gives each word three scores: valence (pleasant or unpleasant), arousal (calm or excited), and dominance (submissive or assertive). Words that aren't in the lexicon inherit their cluster's average.
- **Phonetic fallback.** If a word isn't a real word in either English or German (Hugo Ball's Dada sound-poems are the test case), the system gives it a vowel-driven pitch class and a consonant-driven articulation, and skips the recurrence rule for it.
- **Recurrence table.** Computes the cosine similarity between every pair of words. This table is the central signal the decision layer reacts to.

### Decision layer

Turns features into musical events.

The core is a Markov chain over 72 states: 12 chromatic pitch classes by 6 duration classes (16th, 8th, quarter, dotted-quarter, half, whole). For each cluster, a transition matrix is built from the cluster's centroid using fixed weights for three competing pressures: penalise big pitch jumps, penalise rhythm changes, and bias the chain toward pitch classes that the cluster "prefers". The cluster's pitch preference comes from projecting its centroid through a seeded random matrix and softmaxing the result. No music data is loaded; every entry has a closed-form reason and is reproducible from the seed.

At sampling time, the transition row is reshaped by three forces:

1. **Similarity to the previous word.** When the meanings are close, small intervals get boosted; when they're far apart, larger leaps get boosted. This is what makes "sword/dagger" sound stepwise and "sword/airport" jumpier.

2. **Arousal of the current word.** Excited words push the sampler toward shorter notes; calm words push it toward longer ones.

3. **Recurrence.** This is the signature device. If the current word has high similarity to some earlier word, the chain is pushed toward replaying the note material from that earlier moment. What "replay" means depends on whether the emotional context has shifted:

   - **Emotion roughly the same.** Replay follows a cyclic schedule. The first occurrence is a literal quote. The second is transposed up a perfect fourth. The third is inverted around the previous note. The fourth is transposed down a perfect fifth. The fifth bumps the register up an octave. Then the cycle restarts. Stein's "rose" at positions 0, 3, 6, 9 walks through this schedule monotonically.

   - **Emotion has shifted.** Replay is transformed by the direction of the shift. More pleasant valence shifts the pitch class upward. More excited arousal shortens the duration. More assertive dominance raises the playback velocity.

   How strongly the system pushes toward the replay scales with how strong the recurrence is. A near-perfect match like Stein's "rose" → "rose" gets a strong push and the chain almost always lands on the target. A borderline match between semantic neighbours gets a gentle nudge that the Markov chain can still overrule. The music never literally repeats unless the text does, and the closer the text comes to repeating, the closer the music does too.

The Markov chain runs on every non-phonetic note. The recurrence rule biases its transition row before sampling but never replaces the sampling step itself.

### Score layer

Writes the LilyPond source by hand. Staying close to raw LilyPond (rather than going through Abjad) keeps the output easy to read and inspect. Pitches are clamped softly to the A2–C6 range and hard to A0–C8 (full piano). The score header carries the seed, the model name, and an 8-character config fingerprint, so any rendered PDF can be traced back to a reproducible run.

## How each text feature shapes the music

| Text feature | Musical effect |
|---|---|
| One word | One note |
| Cosine similarity between current and previous word | Interval size — close meaning leans toward small steps, distant meaning toward larger leaps |
| Valence (pleasantness) | Dynamic level on a six-band scale from `\pp` to `\f`; strongly positive or negative valence also shifts register up or down an octave |
| Arousal (excitement) | Duration — excited words lean toward shorter notes |
| Strong text-level recurrence | Markov row pushed toward replay of an earlier word's pitch and duration |
| Same-emotion recurrence | Cyclic motivic development: literal quote, +P4, inversion, −P5, +8va |
| Shifted-emotion recurrence | Replay transposed and re-timed by the direction of the emotional change |
| ALL-CAPS word | `\ff` dynamic with an accent mark |
| Punctuation (`,` `.` paragraph break) | 8th / quarter / half rest after the preceding note |
| Word with no embedding (Dada, OOV) | Phonetic mode: vowel chooses pitch class, first consonant chooses articulation; the recurrence rule is skipped |

## Reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Optional, for emotion-driven dynamics.
python scripts/download_lexicons.py

# Render a text.
python -m semtune --text resources/texts/04_stein_sacred_emily.txt --seed 42 \
                  --out output/sacred_emily/
```

LilyPond must be on `PATH` for the PDF and MIDI. Without it, you still get the `.ly` and `trace.md`.

## Design notes

A rule-based system is the right shape for transparent semantic-to-music mapping. Nothing the system does is hidden behind a learned model. The `trace.md` log makes that explicit: every note has a row, every row names the rule.

What sets this apart from other text-to-music approaches is using the text's own self-similarity as the driver of musical motif recurrence. Story2MIDI, EmotionBox, MINUET, Mel2Word and similar systems map features to notes locally, word by word, or via a learned model's attention. None of them treat text-level recurrence as a first-class signal for musical recurrence.

The chromatic (not diatonic) state space and the closed-form transition matrix (trained on no music corpus) are deliberate choices. The aesthetic target is "new music" rather than common-practice tonality, and synthesising the matrix avoids any copyright exposure on training material.

## Known limitations

- The NRC-VAD lexicons need to be downloaded once for the emotion-driven features to work. Without them, dynamics default to `\mf` throughout.
- With a single-word repetition like Stein's "Rose is a rose is a rose is a rose", short function words (`is`, `a`) also trigger the recurrence rule. The resulting three-note ostinato textures are defensible (the text really does say "is" three times), but raising `theta_quote` thins them out.
- The system is monophonic by design. Adding voices would mean implementing a polyphonic backend, which the abstract `ScoreBackend` interface anticipates.
