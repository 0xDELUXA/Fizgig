# Fizgig v2.13.3 — Pods that boot

## If you deployed the RunPod template, update it to `2.13.3`

Pods created from the public template never started. The image downloaded fine, then the container
died before printing a single line and RunPod restarted it forever.

The cause was one line generating the pod's password. `head` stopped reading after 12 bytes and
closed the pipe; `tr` was still reading `/dev/urandom`, took a `SIGPIPE`, and returned 141 — which
`set -euo pipefail` turned into an immediate exit, one statement before the first log message.

It only ran when no `VNC_PASSWORD` was set, so it survived every test (all of which set one) and
became the default path the moment the public template shipped without a password.

**Change the image tag in your template to `2.13.3` and redeploy.** The entrypoint lives inside the
image, so existing pods can't pick this up.

## Deploy on RunPod, from the Start tab

The Start tab now has a **⚡ Deploy on RunPod** button next to the tip jar, so renting a bigger card
is one click from inside the app.

**Get help on YouTube** is now just **Tutorial** — the play icon already says where it goes, and
these are walkthroughs rather than troubleshooting.

## Better pod instructions

- **Set your own password when you deploy.** The generated one works, but it means fishing a
  credential out of a log that doesn't always render its newest lines. Setting `VNC_PASSWORD`
  yourself skips that entirely.
- **First boot takes a few minutes.** The pod reports itself ready while the image is still
  downloading, so both links appear dead. Now said up front, with the log line that means it's
  genuinely up.
- **Storage is simpler than it looked.** The old advice led with "create a Network Volume", which is
  poor guidance when the regions that have them often don't have the GPU you want. The default
  Volume Disk is fine — the rule that matters is that **stopping a pod keeps everything and
  terminating it doesn't**.
- **`FETCH_MODELS` lists its values** (`krea2`, `klein`, `tools`) instead of naming a variable you
  had to read the source to use.

## Fixes

- The pod's version footer — the one line you're asked to quote in a bug report — was 8pt at 2.54:1
  contrast. Now readable.

---

**Upgrading:** run `update_fizgig.bat`, or `git pull`. On RunPod, set the template's image tag to
`2.13.3`.
