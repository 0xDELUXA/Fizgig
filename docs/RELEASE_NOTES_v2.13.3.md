# Fizgig v2.13.3 — RunPod maintenance

Polish and a startup fix for the RunPod template, plus a quicker route to it from inside the app.

## Running on RunPod? Set your template's image tag to `2.13.3`

A pod startup fix ships in this image. The entrypoint lives inside the image rather than in the
app, so point your template at `2.13.3` and redeploy to pick it up.

## Deploy on RunPod, from the Start tab

The Start tab now has a **⚡ Deploy on RunPod** button next to the tip jar, so renting a bigger card
is one click from inside the app.

**Get help on YouTube** is now just **Tutorial** — the play icon already says where it goes, and
these are walkthroughs rather than troubleshooting.

## Clearer pod instructions

- **Set your own password when you deploy.** Add `VNC_PASSWORD` on the deploy screen and you'll know
  it up front, rather than looking it up afterwards.
- **First boot takes a few minutes** while the image downloads — now said up front, along with the
  log line that means the pod is genuinely ready.
- **Storage is simpler than it looked.** The default Volume Disk is fine; the rule that matters is
  that **stopping a pod keeps everything, terminating it doesn't**. A Network Volume is an optional
  upgrade, not a prerequisite — handy, since the regions offering them don't always have the GPU
  you want.
- **`FETCH_MODELS` now lists its values** — `krea2`, `klein`, `tools`.

## Fixes

- The version line on the pod card — the one you're asked to quote when reporting a problem — is now
  a readable size and contrast.

---

**Upgrading:** run `update_fizgig.bat`, or `git pull`. On RunPod, set the template's image tag to
`2.13.3`.
