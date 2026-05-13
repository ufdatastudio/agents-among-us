# PubApps Deployment Runbook

This runbook covers deploying and operating the public Agents Among Us demo on UF Research Computing's [PubApps](https://docs.rc.ufl.edu/services/web_hosting/) infrastructure. The live URL is **https://agents-among-us.rc.ufl.edu**, served from the `pubufdatastudios1` VM.

## What runs there

A single rootless Podman container, built from `Containerfile.navigator`, managed by systemd via a Quadlet unit. The container runs the Flask frontend in `APP_MODE=navigator`, so only the UF Navigator provider is exposed and `.env` is not auto-loaded. Visitors paste their own UF Navigator API key into the configuration UI.

| Component | Path on VM |
| --- | --- |
| Repo clone | `~/agents-among-us/` |
| Deploy script | `~/agents-among-us/container/pubapps-navigator-deploy.sh` |
| Container build recipe | `~/agents-among-us/Containerfile.navigator` |
| Quadlet unit (generated) | `~/.config/containers/systemd/agents-among-us.container` |
| systemd service name | `agents-among-us.service` (user scope) |
| Container image | `localhost/agents-among-us-navigator:latest` |
| Image storage | `/podman/ufdatastudios/containers/storage` (40 GB local disk) |
| Persistent volumes | `~/agents-among-us/logs` and `~/agents-among-us/frontend/data` (mounted into `/app/logs`, `/app/frontend/data`) |

## Prerequisites

1. A PubApps VM provisioned by RC support, with rootless podman storage already pointing at `/podman` (RC handles this).
2. SSH access via the HiPerGator login node as a jump host:

    ```bash
    ssh -J <ufid>@hpg.rc.ufl.edu:2222 ufdatastudios@pubufdatastudios1
    ```

3. RC support has agreed on the host port (currently `8080`) and configured the nginx reverse proxy from `agents-among-us.rc.ufl.edu` to that port. See [RC support handoff](#rc-support-handoff) for the email template.

## First-time deployment

```bash
ssh -J <ufid>@hpg.rc.ufl.edu:2222 ufdatastudios@pubufdatastudios1
git clone https://github.com/ufdatastudio/agents-among-us.git ~/agents-among-us
cd ~/agents-among-us
./container/pubapps-navigator-deploy.sh setup --port 8080
```

The `setup` action builds the image (~1.7 GB, ~3 min), writes the Quadlet unit, reloads systemd, and starts the service. Linger is enabled so the service survives logout and starts at boot.

Verify locally on the VM:

```bash
curl -i http://localhost:8080/api/health
```

Then verify externally once RC has the proxy live:

```bash
curl -i https://agents-among-us.rc.ufl.edu/api/health
```

## Monitoring

```bash
# From ~/agents-among-us on the VM
./container/pubapps-navigator-deploy.sh status     # systemctl --user status
./container/pubapps-navigator-deploy.sh logs       # podman logs -f (tails forever)
podman stats agents-among-us                       # CPU/memory snapshot
journalctl --user -u agents-among-us -n 200        # systemd journal
df -h /podman                                      # image storage usage
```

The `/api/health` endpoint returns 200 once Flask is bound. The Quadlet's `Restart=on-failure` respawns the container if the process exits non-zero.

## Updating

After pushing changes to `main` on GitHub, on the VM:

```bash
cd ~/agents-among-us
git pull --ff-only
./container/pubapps-navigator-deploy.sh rebuild
```

`rebuild` re-runs `podman build` against the current working tree and restarts the service. Bind-mounted logs and frontend data persist because they live on the host filesystem.

## Stopping or removing

```bash
./container/pubapps-navigator-deploy.sh stop       # stop, keep Quadlet
./container/pubapps-navigator-deploy.sh uninstall  # remove Quadlet; image stays
podman rmi localhost/agents-among-us-navigator:latest  # remove the image
```

## Troubleshooting

**Service won't start.** Read `journalctl --user -u agents-among-us -n 100`. The most common cause is a port collision; run `ss -ltn | grep 8080` on the VM to confirm nothing else is bound.

**Health endpoint times out.** Inspect container logs (`pubapps-navigator-deploy.sh logs`). If Flask exited during import, the `APP_MODE=navigator` gating in `config/app_mode.py` is likely tripping over a missing optional dep.

**Reverse proxy returns 502.** The container is unreachable from the proxy. Confirm `systemctl --user status agents-among-us` reports `active (running)` and that `curl http://localhost:8080/` works on the VM. If both pass, the issue is on RC's nginx side; reply to your support ticket.

**`/podman` filling up.** Old image layers accumulate after many `rebuild` cycles. Reclaim with `podman image prune -f`. The disk is 40 GB.

**SSH master expires mid-session.** The ControlMaster socket on your laptop sometimes drops. Re-run any plain `ssh hpg.rc.ufl.edu` command to re-auth (Duo prompt), then retry.

## RC support handoff

When provisioning a new VM or rotating the deployment, the following details cover everything RC needs.

```
Hi,

Deployment summary for the reverse-proxy config:

  Host:            pubufdatastudios1
  Port:            8080 (HTTP)
  Health endpoint: GET /api/health -> 200 OK
  Process:         rootless podman + systemd Quadlet
                   (~/.config/containers/systemd/agents-among-us.container)
  Public DNS:      agents-among-us.rc.ufl.edu

Thanks,
<name>
```

## Why navigator mode (no GPU)

PubApps VMs do not include GPUs. The `Dockerfile` at the repo root uses `nvidia/cuda:12.8.0-...` as a base, builds an ~11 GB image, and assumes `APP_MODE=full` at runtime. None of that fits a PubApps host.

`Containerfile.navigator` uses `ubuntu:22.04`, installs only the `api` extras, and sets `APP_MODE=navigator`. `config/app_mode.py` then disables `should_load_gpu()`, restricts providers to `{navigator}` only, and skips dotenv autoloading. The result is a ~1.7 GB image suitable for a public, bring-your-own-key demo.
