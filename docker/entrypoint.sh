#!/usr/bin/env bash
# Bring up a virtual screen, put Fizgig on it, and serve it to a browser over noVNC.
#
# Order matters: the screen has to exist before Tkinter starts, and the VNC server has to be
# running before websockify has anything to proxy.
set -euo pipefail

log() { echo "[fizgig] $*"; }

REPO_URL="${FIZGIG_REPO:-https://github.com/shootthesound/Fizgig.git}"
REPO_REF="${FIZGIG_REF:-master}"
APP_DIR="/workspace/Fizgig"
# Shaped for Fizgig rather than for a monitor. Its window is 1360x1124 and its content wraps at
# 760px, so width past ~1400 is dead space — 1600 is enough for the window plus a dialog beside
# it (the Repair Studio pop-out preview is the case that wants room). Height is the axis that
# matters: every extra pixel is one less pixel of in-app scrolling, and 1400 gives ~280 more
# than the window was designed around. A 16:9 desktop spends that budget on width nobody uses.
#
# Set SCREEN_SIZE to taste — taller means less scrolling. Worth knowing: noVNC shows the desktop
# 1:1 by default, so a tall desktop shows more at full size; only if you turn scaling on does it
# shrink to fit, at which point height costs you legibility.
SCREEN="${SCREEN_SIZE:-1600x1400x24}"

mkdir -p /workspace/.cache /workspace/.insightface

# ---------------------------------------------------------------- VNC password
# An unprotected VNC on a public URL is a shell on someone else's GPU. If no password is set we
# generate one and print it, rather than leaving it open — a random password in the logs is
# recoverable; an open port is not fixable after the fact.
if [ -z "${VNC_PASSWORD:-}" ]; then
  VNC_PASSWORD="$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 12)"
  log "No VNC_PASSWORD set — generated one for this pod: ${VNC_PASSWORD}"
  log "Set VNC_PASSWORD in the template to choose your own."
fi
mkdir -p /root/.vnc
x11vnc -storepasswd "$VNC_PASSWORD" /root/.vnc/passwd >/dev/null 2>&1

# ---------------------------------------------------------------- Fizgig source
# Pulled rather than baked, so a new release needs no image rebuild. A pod restart is an update.
if [ -d "$APP_DIR/.git" ]; then
  log "Updating Fizgig in $APP_DIR"
  git -C "$APP_DIR" fetch --depth 1 origin "$REPO_REF" && git -C "$APP_DIR" reset --hard FETCH_HEAD
else
  log "Cloning Fizgig ($REPO_REF)"
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$APP_DIR"
fi

# Top up dependencies in case the checkout is newer than the image. Almost always a no-op, and
# a few seconds when it isn't — far cheaper than making users re-pull 10 GB for a new pin.
log "Checking dependencies"
uv pip install --link-mode=copy --index-strategy unsafe-best-match \
   -r "$APP_DIR/requirements.txt" 2>&1 | tail -2 || log "dependency top-up skipped"

# ---------------------------------------------------------------- persistent paths
# Point the portable output dirs at the volume. Model paths are left alone: fetch_models writes
# those itself, and overwriting them here would stomp a path the user set by hand.
python3 - <<'PY'
import json, os
p = "/workspace/Fizgig/prefs.json"
prefs = {}
if os.path.exists(p):
    try:
        prefs = json.load(open(p, encoding="utf-8"))
    except Exception:
        prefs = {}
for key, path in (("lora_output_dir", "/workspace/output_loras"),
                  ("profiles_dir",    "/workspace/profiles"),
                  ("cache_dir",       "/workspace/cache")):
    prefs.setdefault(key, path)
    os.makedirs(prefs[key], exist_ok=True)
os.makedirs("/workspace/models", exist_ok=True)
json.dump(prefs, open(p, "w", encoding="utf-8"), indent=2)
print("[fizgig] output dirs on /workspace")
PY

# ---------------------------------------------------------------- optional model prefetch
# Off by default. Downloading tens of GB unasked spends the user's money and may fetch the
# family they didn't want — the Preferences button does this on demand, with a progress bar.
if [ -n "${FETCH_MODELS:-}" ]; then
  log "FETCH_MODELS=${FETCH_MODELS} — downloading before launch"
  IFS=',' read -ra FAMS <<< "$FETCH_MODELS"
  ARGS=()
  for f in "${FAMS[@]}"; do ARGS+=(--family "$(echo "$f" | xargs)"); done
  ( cd "$APP_DIR" && PYTHONPATH="$APP_DIR/src" python3 -m fizgig.scripts.fetch_models \
      "${ARGS[@]}" ${FETCH_MODELS_EXTRA:-} ) || log "model fetch incomplete — use the Preferences button"
fi

# ---------------------------------------------------------------- the screen
# A restarted container keeps /tmp, so the previous run's X lock survives and Xvfb refuses to
# start with "Server is already active for display 1" — even though nothing is running. x11vnc
# then has no display, exits, and the container dies with no obvious cause. Stopping and
# starting a pod is routine, so this has to be cleared every boot, not just on first run.
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1

log "Starting virtual display ($SCREEN)"
Xvfb :1 -screen 0 "$SCREEN" -nolisten tcp &
for _ in $(seq 1 50); do xdpyinfo -display :1 >/dev/null 2>&1 && break; sleep 0.2; done

# Tkinter draws Toplevels without decoration and mishandles focus if nothing is managing the
# screen — Fizgig leans on dialogs (Repair Studio pop-out, the token prompt, Problem Images),
# so a window manager is required, not cosmetic.
openbox &
x11vnc -display :1 -forever -shared -rfbauth /root/.vnc/passwd -rfbport 5900 -quiet -bg >/dev/null

log "Serving noVNC on :6080"
websockify --web=/usr/share/novnc 6080 localhost:5900 &

cat <<EOF
[fizgig] ------------------------------------------------------------
[fizgig]  Ready. Open the pod's HTTP port 6080 and connect.
[fizgig]  VNC password: ${VNC_PASSWORD}
[fizgig]
[fizgig]  Models: Preferences -> "Download models for me"
[fizgig]          Krea 2 needs no HuggingFace account; Klein needs a token.
[fizgig] ------------------------------------------------------------
EOF

# Maximise Fizgig once it appears. Its geometry is tuned for a Windows desktop with Segoe UI;
# under DejaVu the tab strip is wider, so "Preferences" truncates at the designed width. Filling
# the screen fixes that and stops a browser-sized viewport being mostly empty desktop. Runs in
# the background because the GUI has to exist first, and the launch below never returns.
(
  for _ in $(seq 1 90); do
    if wmctrl -l 2>/dev/null | grep -q "Fizgig"; then
      wmctrl -r "Fizgig" -b add,maximized_vert,maximized_horz 2>/dev/null && \
        log "window maximised"
      break
    fi
    sleep 1
  done
) &

cd "$APP_DIR"
export PYTHONPATH="$APP_DIR/src"
# exec so Fizgig is PID 1's successor: a container stop signals the app directly, and the
# container's lifetime is the app's lifetime rather than outliving a crashed GUI.
exec python3 lora_trainer_gui.py
