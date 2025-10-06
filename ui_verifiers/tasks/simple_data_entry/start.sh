#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden by environment)
: "${DISPLAY:=:1}"
: "${DISPLAY_GEOMETRY:=1280x800}"
: "${DISPLAY_DEPTH:=24}"
: "${VNC_PASSWORD:=ui-verifiers}"

export USERNAME="${USER:-ui-verifiers}"
export HOME_DIR="${HOME:-/home/ui-verifiers}"

mkdir -p "${HOME_DIR}/.vnc"

# Configure VNC password non-interactively on every start (support both tigervnc and tightvnc variants)
VNC_PASSWD_BIN="$(command -v vncpasswd || true)"
printf "%s\n" "${VNC_PASSWORD}" | "${VNC_PASSWD_BIN}" -f > "${HOME_DIR}/.vnc/passwd"
chmod 600 "${HOME_DIR}/.vnc/passwd"

# Create a minimal xstartup if not present
XSTARTUP_PATH="${HOME_DIR}/.vnc/xstartup"
if [ ! -s "${XSTARTUP_PATH}" ]; then
cat > "${XSTARTUP_PATH}" << 'EOF'
#!/usr/bin/env sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
XRDB_BIN="$(command -v xrdb || true)"
if [ -n "$XRDB_BIN" ] && [ -f "$HOME/.Xresources" ]; then
  "$XRDB_BIN" "$HOME/.Xresources"
fi
# Start Openbox, run the startup command, and keep Openbox in the foreground
dbus-launch --exit-with-session openbox-session &
echo "DISPLAY=${DISPLAY}"
sleep 2
python3 server.py &
wait
EOF
chmod +x "${XSTARTUP_PATH}"
fi

# Clean up any stale locks from previous crashes and ensure X11 socket dir exists
rm -f /tmp/.X*-lock || true
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix
rm -f /tmp/.X11-unix/X* || true

# Ensure ownership (in case the script was run as root previously)
#chown -R "${USERNAME}:${USERNAME}" "${HOME_DIR}/.vnc" || true

# Kill existing server for this DISPLAY if running
if vncserver -list 2>/dev/null | grep -q "${DISPLAY}"; then
  vncserver -kill "${DISPLAY}" || true
fi

# Launch VNC server in the foreground so the container stays alive
exec vncserver "${DISPLAY}" \
  -localhost no \
  -geometry "${DISPLAY_GEOMETRY}" \
  -depth "${DISPLAY_DEPTH}" \
  -fg


