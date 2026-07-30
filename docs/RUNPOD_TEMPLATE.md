# RunPod template README

Paste the section below into the **README** tab of the RunPod template.

Two rules if you edit it later. **This is read before deploying**, so the things that cost money or
lose data go above the fold and everything else links out — the full docs are one click away and do
not need repeating here. And it is **not a minimum-spec pitch**: "Krea 2 fits in 8 GB" belongs in
the project README, where someone is deciding whether Fizgig runs on hardware they already own.
Here they are choosing what to rent, and the appeal is getting a bigger card than they have.

---

# Fizgig — Klein 9B & Krea 2 LoRA Studio

Train, profile, repair and extract LoRAs for **Flux 2 Klein 9B** and **Krea 2** — the full desktop
app in your browser, on whatever GPU you feel like renting.

## Before you deploy

**Attach a Network Volume (100 GB+) at `/workspace`.**

Not the template's *Volume Disk* — that one is deleted when you terminate the pod, taking ~45 GB of
downloaded models with it. A Network Volume outlives any pod, so you download the models once.

## What you get

- **The whole app** — Training, Repair Studio, LoRA the Explorer, LoRA Royale, Profiler, Extract
  and the sample gallery. Not a cut-down web version.
- **Drag-and-drop file transfer** on port 8080 — datasets in, LoRAs out, no terminal
- **One-click model downloads** — Krea 2 needs no HuggingFace account
- **Auto-stop** — optionally shut the pod down when a run finishes, so an overnight finish doesn't
  bill until morning

## Logging in

Both ports ask for a username and password:

| | Port | Username | Password |
|---|---|---|---|
| **Fizgig** | 6080 | `fizgig` | see below |
| **File manager** | 8080 | `admin` | the same one |

**The password is in the pod log.** Fizgig generates a fresh one for every pod and prints it at
startup — open the pod's log and copy it.

**Want your own instead?** On the deploy screen expand **Edit Template → Environment Variables**
and set `VNC_PASSWORD` before launching. Use 12+ characters: shorter ones get zero-padded for the
file manager, and the log tells you what it ended up as.

## First run

1. Connect on **port 6080** and log in
2. **Preferences → ⬇ Download models for me** (Krea 2 ~45 GB, no account needed)
3. Open **port 8080** and drag a dataset folder into `/workspace/datasets`
4. **Start tab → Browse** → pick it, then **Training → Start**

Closing the browser tab does **not** stop training. Fizgig runs on the pod — shut the tab, come
back later, the run is still going.

## Settings

Environment variables — set them on the deploy screen under **Edit Template**.

| Variable | |
|---|---|
| `VNC_PASSWORD` | Desktop *and* file manager. **12+ characters.** |
| `HF_TOKEN` | Klein only. Krea 2 needs nothing. |
| `FETCH_MODELS` | e.g. `tools,krea2` to download at boot instead of in the app. |
| `FIZGIG_REF` | Branch or tag of Fizgig to run. Defaults to `master`, so the app updates itself at every pod start regardless of the image version. |

To enable auto-stop, paste a RunPod API key into **Preferences → RunPod** inside the app. Don't put
one in a template — template variables reach every container deployed from it.

## Storage

```
/workspace/datasets/      your training images (one folder per LoRA)
/workspace/models/        the weights
/workspace/output_loras/  finished LoRAs
```

Everything under `/workspace` persists. Anything outside it is wiped when the pod stops.

Prefer a terminal? `runpodctl` is preinstalled — `runpodctl send <path>` on the pod, then
`runpodctl receive <code>` on your machine. `scp` and `rsync` over SSH work too, and rsync is the
better bet for a large dataset since it resumes.

## More

- [Fizgig on GitHub](https://github.com/shootthesound/Fizgig)
- [Running on a rented GPU — full guide](https://github.com/shootthesound/Fizgig/blob/master/docker/README.md)
