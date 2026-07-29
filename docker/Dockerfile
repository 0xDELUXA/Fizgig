# Fizgig on a headless GPU box (RunPod, Vast, or any Docker host with an NVIDIA GPU).
#
# Fizgig is a Tkinter desktop app: it needs a display, and there is no web UI to expose. So the
# image runs a virtual screen (Xvfb) with a light window manager on it, and streams that screen
# to a browser over noVNC. You click Connect and get the real app, not a cut-down version.
#
# Deliberately NOT in this image:
#   * Fizgig's source     — cloned/pulled at boot, so a new release reaches users without a
#                           10 GB image rebuild and a 10 GB re-pull.
#   * Model weights       — Klein's repos are gated (BFL require each user to accept the licence
#                           personally, which bundling would bypass), and both families together
#                           are ~80 GB, which takes longer to pull as an image layer than to
#                           fetch from HuggingFace directly. The entrypoint downloads on demand.
#
# No CUDA base image: the torch cu128 wheels bundle their own CUDA runtime, so a plain Ubuntu
# base plus the host's driver is all that's needed, and it keeps the image several GB smaller.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISPLAY=:1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev python3-pip \
      python3-tk \
      build-essential \
      git curl ca-certificates \
      xvfb x11vnc openbox novnc websockify x11-utils \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      fonts-dejavu-core fonts-liberation \
      xdg-utils \
    && rm -rf /var/lib/apt/lists/*
# python3-tk is the one that is easy to miss and fatal: without it Tkinter is absent and the
# GUI cannot start at all. build-essential is for insightface, which has no manylinux wheel and
# compiles from source. The libgl/libglib set is what opencv needs on a headless box.

# Dependencies are baked so a pod start is fast; only requirements.txt changes need a rebuild.
# Installed into a venv at /opt/venv rather than system Python so the entrypoint's top-up
# install can't fight Ubuntu's package manager.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip uv \
    && uv pip install --link-mode=copy --index-strategy unsafe-best-match \
         -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

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
