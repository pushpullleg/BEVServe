# GPU container host setup (requires an administrator)

Audited 2026-10-27 on the BEVServe GPU host, read-only:

| Item | State |
| --- | --- |
| Kernel | 6.14.0-37-generic, kernel meta-packages pinned to priority -1 (must stay) |
| NVIDIA driver | 570.211.01 (open kernel module, DKMS), RTX PRO 6000 Blackwell, compute capability 12.0 |
| CUDA | driver supports CUDA 12.8 containers (BEVServe image: CUDA 12.8) |
| Docker engine | Ubuntu `docker.io` 29.1.3, active; `/etc/docker/daemon.json` absent |
| Docker access | socket `root:docker` 0660; the service user is **not** in `docker` |
| Compose / Buildx | not installed (`docker compose` unknown) |
| NVIDIA Container Toolkit | not installed; candidate 1.18.1 from the already-configured Lambda repo |
| Running containers | none |

## Proposed changes

Dry run (`apt-get -s install --no-install-recommends ...`, before `apt-get update`):
6 newly installed, **0 upgraded, 0 removed**, no `linux-*`, `nvidia-driver*`, `nvidia-dkms*` or
`libnvidia-compute*` package:

```
docker-buildx 0.30.1-0ubuntu1~24.04.1          (Ubuntu noble-updates)
docker-compose-v2 2.40.3+ds1-0ubuntu1~24.04.1  (Ubuntu noble-updates)
libnvidia-container1 1.18.1+ds-0lambda0.24.04.1        (Lambda repo)
libnvidia-container-tools 1.18.1+ds-0lambda0.24.04.1   (Lambda repo)
nvidia-container-toolkit-base 1.18.1-0lambda0.24.04.1  (Lambda repo)
nvidia-container-toolkit 1.18.1-0lambda0.24.04.1       (Lambda repo)
```

Commands, in order (no reboot; run while no GPU job is active):

```bash
sudo apt-get update
# Re-check after update: must still show 0 upgraded and no kernel/driver packages.
apt-get -s install --no-install-recommends nvidia-container-toolkit docker-compose-v2 docker-buildx
sudo apt-get install --no-install-recommends nvidia-container-toolkit docker-compose-v2 docker-buildx
sudo nvidia-ctk runtime configure --runtime=docker   # creates /etc/docker/daemon.json
sudo systemctl restart docker                        # no containers are running
sudo usermod -aG docker "$USER"                      # takes effect at next login / newgrp docker
# Verify
uname -r            # 6.14.0-37-generic
nvidia-smi          # driver 570.211.01 unchanged
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Risks

- Membership in `docker` is root-equivalent on this host. Alternative: keep the user out of the
  group and have an administrator run the build and `docker compose` commands.
- `systemctl restart docker` stops any running container (none at audit time).
- Packages come from the Lambda repository already trusted by this host.
- `apt-get update` may change candidate versions; re-run the dry run and stop if any kernel or
  driver package appears.

## Rollback

```bash
sudo gpasswd -d "$USER" docker
sudo rm /etc/docker/daemon.json            # did not exist before
sudo systemctl restart docker
sudo apt-get remove nvidia-container-toolkit nvidia-container-toolkit-base \
  libnvidia-container-tools libnvidia-container1 docker-compose-v2 docker-buildx
```
