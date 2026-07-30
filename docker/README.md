# Running Fizgig on a rented GPU

Fizgig is a desktop app, not a web app. This image gives it a virtual screen and streams that
screen to your browser, so what you get is the whole workbench — Repair Studio, Explorer, Royale,
the sample gallery — rather than a cut-down web version of the trainer.

```
your browser  ──HTTP :6080──▶  KasmVNC (X server + web server in one)
                                   └── openbox + Fizgig
```

KasmVNC rather than the usual Xvfb + x11vnc + noVNC stack: it encodes in **WebP**, multi-threaded,
and drops quality while you drag then restores it when you stop. The plain stack was noticeably
laggy over a link to a rented GPU and had no tuning left client-side. It also **resizes the desktop
to match your browser window**, so there is no screen size to guess at.

## RunPod

**[⚡ Deploy Fizgig on RunPod →](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep)**

One click, pre-set to an RTX 5090 — the cheapest card that clears Fizgig's 32 GB no-block-swap
threshold for Krea 2. Deploying through that link supports Fizgig's development at no extra cost
to you.


### Building your own template

Only if you want your own — the Deploy button above needs none of this. Field names as they appear
in RunPod's *Edit template* dialog:

| Field | Value |
|---|---|
| Template name | `Fizgig` |
| Template type | Pods |
| Compute type | NVIDIA · GPU |
| Container image | `ghcr.io/shootthesound/fizgig:2.13.2` — a **version** tag, see below |
| Container disk | `25` GB |
| Persistent storage | **Volume disk**, `100` GB |
| Persistent storage mount path | `/workspace` |

A **public** template must use Volume Disk, not Network storage — RunPod greys out the Public
toggle otherwise, since a network volume belongs to one account. Deployers pick their own.

**Pin a version, not `:latest`.** RunPod caches images per host, so a mutable tag can serve a stale
one with no way to tell what is running. It costs little here — Fizgig pulls its source from git at
every pod start, so app updates arrive whatever the image tag says, and the image itself only
changes when system packages or Python dependencies do.

Under **Networking configuration → HTTP Ports**, add three. The labels show up in the pod's Connect
menu, so name them:

| Label | Port |
|---|---|
| `Fizgig` | `6080` |
| `File Manager` | `8080` |
| `Mobile` | `8081` |

8081 is reserved for future in-app use. Adding it now costs nothing and saves editing the template
later — pods deployed from an older template would not have it.

No TCP ports, no start command, and no registry authentication — the image is public.

Leave **Environment variables** empty — RunPod hands them to everyone who deploys the template, and
Fizgig generates a per-pod password anyway.

**Pick a Network Volume when you deploy.** The template can only offer a Volume Disk (see above);
the choice is yours at deploy time, and the difference only bites at termination — but it bites
hard:

| | Stop the pod | **Terminate** the pod |
|---|---|---|
| **Volume Disk** (created with the pod) | kept | **deleted with the pod** |
| **Network Volume** (a separate resource) | kept | **kept** |

RunPod's own wording for Volume Disk is "tied to the Pod's lifecycle" — terminate takes it, and
your ~45 GB of models with it. That matters more than it sounds, because **every image update
requires terminating and redeploying**, so on a Volume Disk you re-download the models each time.

Create the Network Volume first (Storage → Network Volumes), then select it on the deploy screen
in place of the template's Volume Disk. Two trade-offs: it is billed per GB/month whether or not a
pod is running, and it is region-locked, which fixes the set of GPUs you can rent.

**Environment variables**

| Variable | Default | What it does |
|---|---|---|
| `VNC_PASSWORD` | *generated* | Password for the browser session **and** the file manager. If unset, one is generated and printed to the pod log — **set your own**. Use **12+ characters**: the file manager rejects anything shorter, and a short password gets silently padded with `0`s (the pod log tells you what it ended up as). |
| `FETCH_MODELS` | *(empty)* | Comma-separated families to download before launch, e.g. `tools,krea2`. Left empty on purpose: pulling tens of GB unasked spends your money, possibly on the family you didn't want. Use the button in Preferences instead. |
| `HF_TOKEN` | — | Only needed for Klein. Krea 2's files aren't gated. |
| `RUNPOD_STOP_API_KEY` | *Optional.* Enables *stop the pod when training finishes*. Normally you paste the key into **Preferences → RunPod** instead, which saves it to your volume — **never put a key in a shared template**, since template variables are handed to every container deployed from it. Must be an **account** key (RunPod → Settings → API Keys); the one RunPod injects into a pod is pod-scoped and 403s on pod-management calls, which is a known RunPod limitation rather than a Fizgig one. |
| `FIZGIG_REF` | `master` | Branch or tag to run. Pin it if you want a fixed version. |
| `SCREEN_W` / `SCREEN_H` | `1600` / `1400` | Only the *starting* size — the desktop resizes to match your browser window, so this rarely matters. |

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

1. Open the pod's HTTP `6080` endpoint and log in.

| | Port | Username | Password |
|---|---|---|---|
| **Fizgig** | 6080 | `fizgig` | see below |
| **File manager** | 8080 | `admin` | the same one |

   The password is in the pod log: by default Fizgig generates a fresh one per pod and prints it
   there. Also logged is the zero-padded version the file manager uses when your password is under 12 characters.

   To choose your own, set `VNC_PASSWORD` on the deploy screen under **Edit Template →
   Environment Variables** before launching.
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
  ghcr.io/shootthesound/fizgig:2.13.2
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
