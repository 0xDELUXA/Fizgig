# Running Fizgig on a rented GPU

[**⚡ Deploy Fizgig on RunPod →**](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep)

The whole app in a browser tab — Training, Repair Studio, LoRA the Explorer, LoRA Royale, Profiler,
Extract and the sample gallery. Nothing to install, and your own GPU stays free while it trains.

*That link is a referral one — it supports Fizgig's development at no extra cost to you.*

---

## Before you deploy

**Create a Network Volume first** (Storage → Network Volumes, 100 GB+), then select it on the deploy
screen.

This is the one choice that costs real time to get wrong. The default *Volume Disk* is **deleted
when you terminate a pod**, taking ~45 GB of downloaded models with it. A Network Volume outlives
any pod, so you download the models once and every future session reuses them. Both survive
*stopping* a pod; only the Network Volume survives *terminating* one.

It's billed per GB/month whether or not a pod is running, and it's region-locked — which fixes
which GPUs you can rent later, so pick a region with the cards you want.

## Which GPU

The link defaults to an **RTX 5090**, and that's a deliberate choice rather than "the newest one".

Fizgig sizes Krea 2's block swap to your VRAM, and **at 32 GB it uses none at all**. Block swap
moves weights over PCIe every step and costs roughly **4× the step time**. When you're billed by the
hour, a card that's slightly dearer but four times faster is much cheaper per finished LoRA.

| | Cards | |
|---|---|---|
| **Best** | RTX 5090 (32 GB), L40S / A6000 (48 GB) | Krea 2 with no block swap |
| **Good value** | RTX 4090, 3090, A5000 (24 GB) | Swaps for Krea 2; ideal for Klein 9B |
| **Smallest worth renting** | 16 GB | Fine, but heavy swap makes it false economy by the hour |

H100 and A100 work but are poor value here — LoRA training never touches 80 GB, and you'd pay
several times more for it.

## Logging in

| | Port | Username | Password |
|---|---|---|---|
| **Fizgig** | 6080 | `fizgig` | see below |
| **File manager** | 8080 | `admin` | the same one |

**The password is in the pod log.** Fizgig generates a fresh one for every pod and prints it at
startup — open the pod's log and copy it.

**Want your own?** On the deploy screen expand **Edit Template → Environment Variables** and set
`VNC_PASSWORD` before launching. Use 12+ characters: shorter ones get zero-padded for the file
manager, and the log tells you what it ended up as.

## First run

1. Connect on **port 6080** and log in
2. **Preferences → ⬇ Download models for me** — Krea 2 is ~45 GB and needs no HuggingFace account;
   Klein is gated by Black Forest Labs and will ask for a free token
3. Open **port 8080** and drag a dataset folder into `/workspace/datasets`
4. **Start tab → Browse** → pick it, then **Training → Start**

Closing the browser tab does **not** stop training. Fizgig runs on the pod — shut the tab, come back
later, the run is still going.

## Getting files in and out

**Port 8080** is a file manager rooted at `/workspace`. Drag a dataset folder in from your desktop,
download finished LoRAs from `output_loras/` the same way. No terminal, no SSH keys.

Prefer a terminal? `runpodctl` is preinstalled — `runpodctl send <path>` on the pod, then
`runpodctl receive <code>` on your machine. `scp` and `rsync` over SSH work too, and rsync is the
better bet for a large dataset since it resumes.

## Where things live

```
/workspace/datasets/      your training images (one folder per LoRA)
/workspace/models/        the weights
/workspace/output_loras/  finished LoRAs
```

Everything under `/workspace` persists. Anything outside it is wiped when the pod stops.

## Stopping the pod when a run finishes

A rented GPU bills by the hour, so a run that ends at 4am keeps billing until you notice.
**Preferences → RunPod → Stop this pod when a training run finishes** fixes that. You get a
two-minute countdown you can cancel, and it never fires after a Pause, a Stop or a failure — those
are exactly the times you want the machine alive.

It needs an API key, pasted into that same panel. Make one at **RunPod → Settings → API Keys**; the
key RunPod gives a pod automatically is pod-scoped and cannot stop pods, which is a RunPod
limitation rather than a Fizgig one. It's stored on your volume, not in any template.

## Settings

Environment variables, set on the deploy screen under **Edit Template**:

| Variable | |
|---|---|
| `VNC_PASSWORD` | Desktop *and* file manager. 12+ characters. |
| `HF_TOKEN` | Klein only. Krea 2 needs nothing. |
| `FETCH_MODELS` | e.g. `tools,krea2` to download at boot instead of in the app. |
| `FIZGIG_REF` | Branch or tag of Fizgig to run. Defaults to `master`, so the app updates itself at every pod start. |

## If something's wrong

The **pod log** is the first place to look — every step is narrated there, including storage and the
login details.

- **Storage shows ~25 GB** — the volume didn't mount. Check the mount path is `/workspace`.
- **Downloads fail with "no space left"** — same cause: models are landing on container disk.
- **A restart lost your models** — a Volume Disk was used instead of a Network Volume.

Fizgig's own version and the image's are both shown in **Preferences → RunPod**; quote both if you
report a problem, since the app updates itself independently of the image.
