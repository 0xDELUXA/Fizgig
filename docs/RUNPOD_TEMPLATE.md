# RunPod template README

Paste the section below into the **README** tab of the RunPod template. It's what people read
*before* they deploy, so it leads with what they get rather than how it's built.

Framing note for whoever edits this later: this is **not** a minimum-spec pitch. "Krea 2 fits in
8 GB" belongs in the project README, where someone is deciding whether Fizgig runs on hardware they
already own. Here the reader has no such constraint — they're choosing a card to rent, and the
appeal is getting *more* GPU than they have, or keeping their own machine free while it trains.

---

# Fizgig — Klein 9B & Krea 2 LoRA Studio

Train, profile, repair and extract LoRAs for **Flux 2 Klein 9B** and **Krea 2** — the full desktop
app in your browser, on whatever GPU you feel like renting.

Pick a big card for an afternoon instead of buying one, and leave your own machine free while it
trains.

## What you get

- **The whole app, not a cut-down web version** — Training, Repair Studio, LoRA the Explorer,
  LoRA Royale, Profiler, Extract, and the sample gallery
- **Drag-and-drop file transfer** on port 8080 — datasets in, finished LoRAs out, no terminal
- **One-click model downloads** — Preferences → *Download models for me*. Krea 2 needs no
  HuggingFace account
- **Persistent storage** — models, datasets, caches and LoRAs live on your volume and survive
  stopping the pod, so you download the models once
- **Stop-when-finished** — optionally shut the pod down after a run completes, so an overnight
  finish doesn't bill until morning

## First run

1. Connect on **port 6080**. Log in with username `fizgig` and the `VNC_PASSWORD` you set
   (both are in the pod log).
2. **Preferences → ⬇ Download models for me.** Krea 2 is ~32 GB and needs no account; Klein is
   gated by Black Forest Labs and will ask for a free HuggingFace token.
3. Open **port 8080**, log in as `admin` with the same password, and drag a dataset folder into
   `/workspace/datasets`.
4. **Start tab → Browse →** pick your folder. Then Training → Start.

Closing the browser tab does **not** stop training — Fizgig runs on the pod, so you can shut the
tab and come back later.

## Settings

| Variable | Notes |
|---|---|
| `VNC_PASSWORD` | Used for both the desktop and the file manager. **12+ characters** — shorter ones get padded and the pod log tells you what to. |
| `HF_TOKEN` | Only for Klein. Krea 2 needs nothing. |
| `FETCH_MODELS` | e.g. `tools,krea2` to download before launch. Left empty by default so you choose. |
| `FIZGIG_REF` | Branch or tag to run. Defaults to `master`, which updates on every pod start. |

**Give it a volume of 100 GB+ mounted at `/workspace`.** Without one, models land on the container
disk and are wiped when you stop the pod — you'd re-download 32 GB every session. The pod log
reports what it found on startup and warns if that looks wrong.

## Storage layout

```
/workspace/datasets/      your training images (one folder per LoRA)
/workspace/models/        the weights
/workspace/output_loras/  finished LoRAs
```

## Links

- [Fizgig on GitHub](https://github.com/shootthesound/Fizgig)
- [Full documentation](https://github.com/shootthesound/Fizgig#readme)
