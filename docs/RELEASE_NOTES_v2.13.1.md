# Fizgig v2.13.1 — Train on a GPU you don't own

## Fizgig runs on rented hardware

Fizgig now ships as a ready-made cloud image. Deploy it on a rented GPU and you get the **whole
app** in a browser tab — Training, Repair Studio, LoRA the Explorer, LoRA Royale, Profiler, Extract
and the sample gallery. Not a cut-down web version; the actual application.

Use it to train on a far bigger card than the one in your machine, or to keep your own GPU free
while a run happens somewhere else.

- **Drag-and-drop file transfer** — datasets in, finished LoRAs out, no terminal
- **One-click model downloads** — Krea 2 needs no HuggingFace account
- **Persistent storage** — download the models once; every future session picks up where you left off
- **Stop when finished** — optionally shut the machine down after a run completes, so an overnight
  finish doesn't bill until morning
- **Closing the browser doesn't stop training** — it runs on the machine, not in your tab

There's a new **RunPod section in Preferences**: on a rented machine it's the control panel, and on
your desktop it explains the option and links to the guide.

**[Running Fizgig on a rented GPU →](https://github.com/shootthesound/Fizgig/blob/master/docker/README.md)**

## Readable text

Users reported the Preferences tab as grey-on-grey, and measuring it they were right — the
explanatory text scored **2.54:1** contrast, which fails accessibility guidelines even for large
text. It's now off-white at **8.64:1** and a point larger, across **every tab**, since the same
problem was everywhere and Preferences was just the most text-heavy place to notice it.

The **scrollbar** had the same issue in a worse form: the part you drag was 1.06:1 against its own
track — effectively invisible. It's now the same blue as the selected tab.

## Fixes

- **Training could fail silently to start.** Fizgig launched training with a hardcoded path to its
  bundled virtual environment. If yours lives anywhere else — conda, a system install — the process
  never started and the run stopped dead after "starting cache preparation" with nothing in the
  console explaining why.
- **Folders with `[square brackets]` in the path** no longer break caption discovery.
- **Low disk space** is now checked before a run starts, so you find out before four hours in.

## Since 2.13.0

The pod image now reports its own version. Preferences → RunPod shows the image version alongside
the app's git commit — they differ on purpose, since the image is pinned in your template while
Fizgig updates itself at every pod start, and a bug report needs both numbers.

---

**Upgrading:** run `update_fizgig.bat`, or `git pull`.
