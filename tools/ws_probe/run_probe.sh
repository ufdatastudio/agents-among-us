#!/bin/bash
# Swaps the demo for the WebSocket probe for at most 6 minutes, then starts the demo again.
restore() {
  podman rm -f ws-probe >/dev/null 2>&1
  systemctl --user start agents-among-us.service
  echo "demo started again at $(date +%T)"
}
trap restore EXIT
systemctl --user stop agents-among-us.service
echo "demo stopped at $(date +%T)"
timeout 360 podman run --rm --name ws-probe -p 8080:8080 \
  -v "$HOME/ws_probe:/probe:ro,Z" --entrypoint python \
  localhost/agents-among-us-navigator:latest /probe/ws_probe_server.py --port 8080 --ttl 330
echo "probe finished at $(date +%T)"
