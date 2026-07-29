# Fizgig on a headless GPU box (RunPod, Vast, or any Docker host with an NVIDIA GPU).
#
# Fizgig is a Tkinter desktop app: it needs a display, and there is no web UI to expose. So the
# image runs a virtual screen (Xvfb) with a light window manager on it, and streams that screen
# to a browser over noVNC. You click Connect and get the real app, not a cut-down version.
#
# Deliberately NOT in this image:
#   * Fizgig's source     — cloned/pulled at boot, so a new release reaches users without a
#                           large image rebuild and re-pull.
#   * Model weights       — Klein's repos are gated (BFL require each user to accept the licence
#                           personally, which bundling would bypass), and both families together
#                           are ~80 GB, which takes longer to pull as an image layer than to
#                           fetch from HuggingFace directly. The entrypoint downloads on demand.
#
# No CUDA base image: the torch cu128 wheels bundle their own CUDA runtime, so a plain Ubuntu
# base plus the host's driver is all that's needed, and it keeps the image several GB smaller.

# ---------------------------------------------------------------------------- builder
# Compiles what needs compiling (insightface ships no manylinux wheel) and resolves the whole
# dependency tree into a self-contained venv. None of the toolchain reaches the final image.
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
# The cache purge has to happen in THIS layer: a later RUN would delete the files but the layer
# holding them stays in the image. uv keeps its own wheel cache and ignores PIP_NO_CACHE_DIR,
# which is how a first build shipped 7.8 GB of wheels alongside the 7.8 GB venv they installed.
RUN pip install --upgrade pip uv \
    && uv pip install --link-mode=copy --index-strategy unsafe-best-match \
         -r /tmp/requirements.txt \
    && rm -rf /root/.cache /tmp/requirements.txt

# ---------------------------------------------------------------------------- runtime
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISPLAY=:1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv \
      python3-tk \
      git curl ca-certificates \
      xvfb x11vnc openbox novnc websockify x11-utils wmctrl \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      fonts-dejavu-core fonts-liberation fontconfig unzip \
      xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Fizgig's whole layout is measured in Segoe UI, which does not exist on Linux — Tk silently
# substitutes DejaVu Sans, which renders the same string ~20% wider (321px vs 268px measured).
# That is not cosmetic: every wraplength, column width and window geometry in the app is tuned to
# the narrower metrics, which is why the tab strip truncated "Preferences" before this.
#
# Selawik is Microsoft's own metric-compatible open substitute for Segoe UI (SIL OFL), so aliasing
# it fixes the measurements everywhere at once with no changes to Fizgig itself.
RUN curl -fsSL https://github.com/microsoft/Selawik/releases/download/1.01/Selawik_Release.zip \
      -o /tmp/selawik.zip \
    && mkdir -p /usr/share/fonts/truetype/selawik \
    && unzip -j -o /tmp/selawik.zip '*.ttf' -d /usr/share/fonts/truetype/selawik \
    && rm /tmp/selawik.zip \
    && printf '%s\n' \
      '<?xml version="1.0"?>' \
      '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">' \
      '<fontconfig>' \
      '  <alias binding="same">' \
      '    <family>Segoe UI</family>' \
      '    <accept><family>Selawik</family></accept>' \
      '  </alias>' \
      '</fontconfig>' \
      > /etc/fonts/conf.d/99-segoe-selawik.conf \
    && fc-cache -f
# python3-tk is the one that is easy to miss and fatal: without it Tkinter is absent and the GUI
# cannot start at all. The libgl/libglib set is what opencv needs on a headless box. No
# build-essential here — it was only ever needed to compile insightface in the builder.

COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"

# Everything the user creates lives here so a RunPod network volume keeps it across pods:
# models, datasets, output LoRAs, latent cache, and the HuggingFace/InsightFace caches — which
# would otherwise re-download Florence-2 and the face model on every single boot.
ENV FIZGIG_HOME=/workspace \
    HF_HOME=/workspace/.cache/huggingface \
    INSIGHTFACE_HOME=/workspace/.insightface \
    XDG_CACHE_HOME=/workspace/.cache
WORKDIR /workspace

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# noVNC over HTTP — the only port that needs exposing. 5900 stays internal on purpose: raw VNC
# is unencrypted, and the host's HTTPS proxy is what makes this safe to reach over the internet.
EXPOSE 6080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
