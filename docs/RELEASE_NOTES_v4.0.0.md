# Fizgig v4.0.0 — video, sound, and voices

MiniMax H3 is an omni model — it generates video and audio together — and until now Fizgig
could only feed it photographs. This release completes the picture. Train on **short video
clips**, on **their sound**, on **voice recordings alone** (plain audio files — wav, mp3,
flac, m4a) — and crucially, **all of it at once: photos, clips and voice recordings in one
folder train one LoRA in one run.**

The strict part — clips and voice segments have to be exactly on H3's spec — is handled by
**Gizmo**, a new prep tool that ships with Fizgig. It cuts clips from any footage, cuts voice
segments from any recording, and includes a push-to-record studio that builds a voice dataset
from nothing but a microphone and ten minutes of reading.

There's a second release hiding underneath: **16 GB and 24 GB cards now train H3 on the
accurate int8 base instead of 4-bit**, thanks to a redesign of block swap contributed by
[@rintic-13](https://github.com/rintic-13) that makes it several times faster.

## How do I…

**…train on video clips?** Cut them with **Gizmo** (launch it from the Image Prep tab, or the
*Launch Gizmo* .bat in Fizgig's folder) — it exports clips already on H3's spec — then drop
them into the training folder next to your images, and caption them on Fizgig's
**Captions tab** exactly like a photo (each clip shows a frame from its middle there).
**Photos, clips and voice recordings all train together in the same folder** — no settings, no
separate runs. (Clips from elsewhere work too, if they're exactly on spec: 24 fps, one of
eight lengths, /32 dimensions, 32 kHz audio.)

**…make clips from my footage?** Open Gizmo (from the Image Prep tab, or the *Launch Gizmo*
.bat), drop a video on it, scrub to a moment, pick a length, *Add to queue* — repeat for every
section you want, then *Export queue*. Crop to the subject so every token goes on what you
actually want learned.

**…chop a long video automatically?** Gizmo's **✂ Auto-chop** scans the whole source for scene
cuts and offers every segment as a thumbnail — click to keep or skip, double-click to inspect,
and the keepers join the export queue. No clip ever straddles a cut.

**…train a voice from a recording?** Gizmo's **Voice** tab: open any audio file — or a video,
to use just its soundtrack — mark segments on the waveform, caption them (describe only the
sound; **Transcribe** appends the spoken words), export. Segments come out sample-exact with
their caption `.txt` beside them, ready to drop into the training folder. Voice captions are
written here in Gizmo, not on the Captions tab — describing a voice means hearing it, and the
Captions tab points you back here when it sees audio files.

**…record a voice dataset from scratch?** Voice tab → **🎙 Record**. Gizmo prompts a sentence
to read and a delivery style — rotating both take by take, so tonal range arrives on its own —
you hold the button (or the **R** key) while you speak, and every release lands as a take with
its caption already written, loaded into the editor ready to queue.

**…keep a clip's sound out of training?** Mute it — a per-clip toggle in Gizmo that adds
`_mute` to the filename, so you can also change your mind later by renaming. A muted clip
trains its video normally.

**…train photos, clips and a voice into one LoRA?** Put them all in the same folder — one
trigger word, one run, any mix. If one side of the dataset is much smaller than the other, the
new **Finish one category early** row on the Training tab lets voice (or photos & clips)
finish at a chosen epoch while the rest trains on.

**…set it up?** One extra model file: the **audio VAE** (~605 MB), on its own Preferences row
with a download link. Leave it blank and clips simply train silent; it's required only once
the folder contains voice recordings.

## Gizmo — the clip and voice prep tool

Fizgig refuses off-spec media rather than quietly transcoding it, because silent fixes make
two identical-looking datasets train differently. Gizmo is the other half of that deal: a
separate app (it opens in under a second — no torch, no CUDA) that turns whatever you have
into files Fizgig accepts.

**Video clips:** open any footage — any format, frame rate or size — and cut to-spec clips
from it. First-and-last-frame previews before you commit, frame-accurate stepping, crop to the
subject with shape locks (1:1, 16:9, 9:16…), per-clip sound-or-mute, real-time or slow-motion
from high-frame-rate sources, and a mark-everything-then-export-once queue. The preview
follows the playhead live while you drag. Gizmo also tells you which clip lengths your card
can actually train, at which megapixels, before you cut anything.

**Voice segments:** a waveform editor for cutting training segments out of any recording.
Zoom rides the mouse wheel, segments snap to H3's allowed lengths, space and J-K-L work like
an edit suite, and Whisper transcription (one-time ~150 MB download) appends the spoken words
to your caption. Exports are sample-exact — the trainer's strict duration check always passes.

**The recorder:** push-to-record, in a card on the Voice tab — your takes survive leaving and
re-entering record mode. Prompted sentences span short interjections to lines that fill the
longest slot, rotating between plain, pangram, literary, silly and dark; the delivery style
(cheerfully, wearily, quietly…) rolls at random after every take, visible in a dropdown you
can override — or switch off in ⚙ settings. The mic rolls continuously, so push-to-record
clips nothing: a quarter-second before your press is already in the take.

## 16 GB and 24 GB cards: int8, streamed

Block swap used to round-trip every swapped block between GPU and CPU. But the base model is
*frozen* during LoRA training — nothing about it changes — so the return leg was pure waste.
[@rintic-13](https://github.com/rintic-13) proposed and prototyped a one-way design: blocks
stream host-to-GPU into a small ring of buffers, the next block prefetching while the current
one computes. As promised on [#73](https://github.com/shootthesound/Fizgig/issues/73): this is
his speedup, and 16 GB users get the biggest share of it.

Measured at the same swap depth on the same card, the streamed path is **6.4× faster** than
the old round-trip (rintic-13 measured 2.7× on a 5060 Ti with his prototype). That changes
what the Auto planner picks: swap is now cheap enough that **16 GB and 24 GB cards get the
int8 base** — the checkpoint's own storage, ~0.17% error — where they previously fell back to
4-bit (~9% error the LoRA had to spend capacity correcting):

| Free VRAM | What Auto does now |
|---|---|
| ~30 GB | int8, no block swap |
| ~22 GB | int8, ~14 blocks streamed |
| ~15 GB | int8, ~36 blocks streamed |
| ≤12 GB | 4-bit, as before |

Alongside it, **text-encoder caching now genuinely fits a 16 GB card** — the nvfp4 encoder's
dequantization was rebuilt to run in bounded chunks, removing the out-of-memory (and the
"caching appears frozen" symptom) that 16 GB users hit at the start of a run. Both paths were
validated end-to-end on a hard-capped 16 GB budget, through full runs with previews cycling.

## Finish one category early

Mixed datasets are rarely balanced — thirty photos and two hundred voice takes, or the
reverse. The **Finish one category early** row (Training tab, visible when the dataset is
mixed) takes a category, an epoch, and a mode: **anchor at 10% LR** keeps the finished
category gently in the loop so the shared adapter doesn't drift away from it (recommended), or
**stop completely** skips its steps for speed. Recorded in the LoRA's metadata.

## Voices train best at Likeness and Style

Tested head-to-head on the same voice dataset: **Likeness and Style** converges much faster
and sounds better than *Model default* — identity lives at the clean end of the noise
schedule for voices just as it does for faces. Fizgig now says so on the Training tab whenever
it sees voice recordings in the dataset, and the sample gallery header shows when a dataset is
audio-only.

## Also in this release

- **LoRA names that can't become filenames are refused before the run starts**, with a plain
  message, instead of failing at the first save — reported by
  [@ioritree](https://github.com/ioritree) ([#70](https://github.com/shootthesound/Fizgig/issues/70)).
- **The VRAM warning names the teacher when the teacher is the cause** — diagnosis by
  [@volnodumcev](https://github.com/volnodumcev) ([#71](https://github.com/shootthesound/Fizgig/issues/71)).
- **A clip's cache is never a deleted file's leftovers** — stale cache entries from removed or
  renamed clips are detected and skipped, per file, with a console note.
- **Preferences model-path sections fold up**, and the missing-paths badge says which model
  family it's talking about.
- **Using the Turbo LoRA in ComfyUI? Skip its custom sampler** — current ComfyUI samples H3
  audio cleanly with stock Euler, and the community consensus is 8 steps. Details in the
  README.

## Upgrading

Nothing to do — model paths, datasets, caches and presets are untouched. If you want sound:
add the audio VAE (~605 MB) on its new Preferences row. Everything else is optional too;
stills-only training works exactly as before.
