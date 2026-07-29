# Running Fizgig on a rented GPU

Fizgig is a desktop app, not a web app. This image gives it a virtual screen and streams that
screen to your browser, so what you get is the whole workbench — Repair Studio, Explorer, Royale,
the sample gallery — rather than a cut-down web version of the trainer.

```
your browser  ──HTTP :6080──▶  noVNC  ──▶  x11vnc  ──▶  Xvfb :1 (1920x1080)
                                                            └── openbox + Fizgig
```

## RunPod

**Template settings**

| Field | Value |
|---|---|
| Container image | `ghcr.io/shootthesound/fizgig:latest` |
| Container disk | 25 GB |
| Volume disk | **100 GB+**, mounted at `/workspace` |
| Expose HTTP ports | `6080, 8080` |
| Expose TCP ports | *(none)* |

**Use a network volume.** Without one, every pod start re-downloads tens of GB of models and you
pay GPU time to watch it happen. With one, it's a one-off, and your datasets, LoRAs and caches
survive between pods. Two things to know: network storage is billed per GB/month, and volumes are
region-locked, which limits which GPUs you can rent.

**Environment variables**

| Variable | Default | What it does |
|---|---|---|
| `VNC_PASSWORD` | *generated* | Password for the browser session **and** the file manager. If unset, one is generated and printed to the pod log — **set your own**. Use **12+ characters**: the file manager rejects anything shorter, and a short password gets silently padded with `0`s (the pod log tells you what it ended up as). |
| `FETCH_MODELS` | *(empty)* | Comma-separated families to download before launch, e.g. `tools,krea2`. Left empty on purpose: pulling tens of GB unasked spends your money, possibly on the family you didn't want. Use the button in Preferences instead. |
| `HF_TOKEN` | — | Only needed for Klein. Krea 2's files aren't gated. |
| `FIZGIG_REF` | `master` | Branch or tag to run. Pin it if you want a fixed version. |
| `SCREEN_SIZE` | `1600x1400x24` | Virtual screen size. Shaped for Fizgig, not for a monitor: its content wraps at 760px so width past ~1400 is wasted, while **height is what reduces in-app scrolling**. Raise the height if you want to scroll less (`1600x1800x24`); noVNC shows the desktop 1:1 by default, so taller means more content at full size. |

### Getting files in and out

Port **8080** is a drag-and-drop file manager ([filebrowser](https://filebrowser.org/)) rooted at
`/workspace`. Log in as **`admin`** with your `VNC_PASSWORD`. Drag a dataset folder from your desktop
into a browser tab, and download finished LoRAs from `output_loras/` the same way — no terminal, no
SSH keys, no CLI.

If you'd rather use a terminal, **`runpodctl`** is preinstalled:

```bash
# on the pod, to send a finished LoRA to your machine
runpodctl send /workspace/output_loras/mylora.safetensors
# then on your machine, with the code it prints
runpodctl receive <code>
```

`scp` and `rsync` over RunPod's SSH also work, and rsync is the better choice for a large dataset
since it resumes and syncs incrementally.

**First run**

1. Open the pod's HTTP `6080` endpoint and enter the VNC password from the pod log.
2. Fizgig is already running. Go to **Preferences → ⬇ Download models for me**.
   Krea 2 needs no HuggingFace account; Klein will ask for a token.
3. Point the **Start** tab at a dataset folder and train.

Models land in `/workspace/models`, LoRAs in `/workspace/output_loras`, both on the volume.

## GPU sizing

Krea 2 trains on **8 GB** with everything on Auto and batch size 1, so the cheap end of the GPU
list is genuinely usable. 10–12 GB gives headroom to raise batch size or resolution. Klein 9B wants
16 GB. See the [VRAM guidance](../README.md#vram-guidance).

## Vast.ai

Same image. Set the Docker image, expose port `6080`, mount storage at `/workspace`, and pass the
same environment variables. Vast's on-start script equivalent needs nothing extra — the entrypoint
handles everything.

## Running it locally

Any machine with Docker and an NVIDIA GPU:

```bash
docker run --gpus all -p 6080:6080 \
  -v fizgig-data:/workspace \
  -e VNC_PASSWORD=changeme \
  ghcr.io/shootthesound/fizgig:latest
```

Then open <http://localhost:6080/vnc.html>.

## Building it yourself

```bash
docker build -f docker/Dockerfile -t fizgig .
```

The image holds system packages and pip dependencies only. **Fizgig's source is cloned at boot**,
not baked in — so a new release reaches users on their next pod start without an image rebuild or a
10 GB re-pull. Rebuild only when `requirements.txt` changes; CI does this automatically on release
tags.

Model weights aren't baked in either. Klein's repos are gated because Black Forest Labs require
each user to accept the licence personally, and shipping those weights in a public image would
bypass that. Both families together are also ~80 GB — slower to pull as an image layer than to
fetch from HuggingFace directly.

## Security

The VNC session is a full desktop on your pod: anyone who reaches it can use the GPU and read the
volume. Always set `VNC_PASSWORD`. Port `5900` (raw, unencrypted VNC) is deliberately **not**
exposed — only `6080`, which the host serves over HTTPS.
