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

The `setup` action builds the image (~1.3 GB, ~3 min), writes the Quadlet unit, reloads systemd, and starts the service. Linger is enabled so the service survives logout and starts at boot.

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

`rebuild` re-runs `podman build --pull=newer --no-cache` against the current working tree and restarts the service. Bind-mounted logs and frontend data persist because they live on the host filesystem.

## Security updates

RC emails a ticket when its scanner finds fixable vulnerabilities on the VM. The deadlines count from detection: critical within 14 days, CISA Known Exploited Vulnerabilities by the catalog due date, and everything else within 30 days. If a container cannot be patched in time, policy requires shutting it down.

Two layers need patching. We own the container image. RC administers the VM operating system, and the `ufdatastudios` account has no sudo.

### Container image

A container is not patched by editing it. The Quadlet unit starts it from `localhost/agents-among-us-navigator:latest`, so changes made with `apt` or `pip` inside a running container disappear on the next restart. Fix the image and replace the container.

1. Find where the findings are. Most are Python packages pinned in `uv.lock`. Upgrade only the flagged packages and commit the result.

    ```bash
    uv lock --upgrade-package urllib3 --upgrade-package cryptography
    ```

    Avoid a full `uv lock --upgrade`. It can change scikit-learn and xgboost, and the pickled classifiers in `results/classifiers/models` depend on those versions. Pillow 13 and later require Python 3.11 or newer, which is why the image uses Python 3.12 (`PYTHON_VERSION` in `Containerfile.navigator`).

2. On the VM, pull the change, keep a rollback tag, and build under a temporary tag so the live service keeps running.

    ```bash
    cd ~/agents-among-us
    git pull --ff-only
    podman tag localhost/agents-among-us-navigator:latest localhost/agents-among-us-navigator:rollback
    podman build --pull=newer --no-cache -t localhost/agents-among-us-navigator:patched -f Containerfile.navigator .
    ```

3. Start the new image on a spare port and check it.

    ```bash
    podman run -d --name aau-verify -e PORT=18080 -p 18080:18080 localhost/agents-among-us-navigator:patched
    curl -fsS http://localhost:18080/api/health
    podman rm -f aau-verify
    ```

4. Check for games in progress. A restart ends them.

    ```bash
    podman top agents-among-us pid etime args | grep main.py
    find ~/agents-among-us/logs/<game_dir> -type f -mmin -60   # no output means no log writes in the last hour
    ```

5. Replace the running container.

    ```bash
    podman tag localhost/agents-among-us-navigator:patched localhost/agents-among-us-navigator:latest
    systemctl --user restart agents-among-us
    curl -fsS http://localhost:8080/api/health
    ```

    The `rebuild` action in the deploy script does steps 2 and 5 together without the staging and rollback tag.

6. To roll back, retag the old image and restart.

    ```bash
    podman tag localhost/agents-among-us-navigator:rollback localhost/agents-among-us-navigator:latest
    systemctl --user restart agents-among-us
    ```

7. Scan the deployed image. The VM has no scanner, so stream the image to a machine with Docker and run Trivy there. The expected result is 0 findings.

    ```bash
    ssh -J <ufid>@hpg.rc.ufl.edu:2222 ufdatastudios@pubufdatastudios1 \
      'podman save localhost/agents-among-us-navigator:latest' | docker load
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:latest \
      image --platform linux/amd64 --severity HIGH,CRITICAL --ignore-unfixed \
      localhost/agents-among-us-navigator:latest
    ```

8. After a week without problems, remove the old tags and layers with `podman rmi localhost/agents-among-us-navigator:rollback` and `podman image prune -f`.

Rebuild at least monthly even without a ticket, so the Ubuntu base and apt packages stay current.

### Host operating system

The VM runs RHEL 9. Applying its updates needs root, so request them from RC by replying to the vulnerability ticket or emailing RC support. The account can list what is pending without sudo:

```bash
dnf check-update --security
uname -r                    # running kernel
rpm -q kernel --last        # installed kernels, newest first
```

Kernel, glibc, and systemd updates take effect only after a reboot, and the reboot ends any game in progress. A person with sudo runs:

```bash
sudo dnf upgrade --security -y
sudo reboot
```

Linger is enabled for `ufdatastudios`, so the Quadlet service starts again at boot without a login. After the VM is back, confirm the update and the service:

```bash
uptime -s                                 # new boot time
uname -r                                  # new kernel version
systemctl --user status agents-among-us
curl -fsS http://localhost:8080/api/health
```

If the service does not start, follow the first item under [Troubleshooting](#troubleshooting). Message for RC:

```
Hi,

pubufdatastudios1 (ufdatastudios) has pending RHEL 9 security updates,
including a newer kernel. The account has no sudo, so please run
"dnf upgrade --security" and reboot at a convenient time. Rebooting ends
running games, so a short notice is appreciated. The agents-among-us
service starts at boot (linger enabled). Health check afterwards:
GET /api/health -> 200 OK.

Thanks,
<name>
```

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

`Containerfile.navigator` uses `ubuntu:22.04` with Python 3.12, installs only the `api` extras, and sets `APP_MODE=navigator`. `config/app_mode.py` then disables `should_load_gpu()`, restricts providers to `{navigator}` only, and skips dotenv autoloading. The result is a ~1.3 GB image suitable for a public, bring-your-own-key demo.
