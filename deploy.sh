#!/usr/bin/env bash
# Ship StackIt to localgeek.jp. Code only — nginx/systemd config lives in
# ~/code/infra and is pushed with `lg push --apply`.
#
#   ./deploy.sh              code + deps, restart
#   ./deploy.sh --with-net   also copy the trained net (it is gitignored, so it
#                            only reaches the server this way)
#
# The net that ships is $NET, default checkpoints/best.pt. Training arms live in
# their own dirs, so point NET at the arm you actually want live, e.g.
#   NET=checkpoints_qblend/best.pt ./deploy.sh --with-net
# It always lands at checkpoints/best.pt on the server, which is where
# server.py looks (Config().ckpt_dir).
set -euo pipefail

HOST="root@167.172.94.195"
PATH_REMOTE="/opt/stackit/app"
VENV="/opt/stackit/venv"

WITH_NET=0
[[ "${1:-}" == "--with-net" ]] && WITH_NET=1
NET="${NET:-checkpoints/best.pt}"

cd "$(dirname "$0")"

echo "==> syncing code to $HOST:$PATH_REMOTE"
# checkpoints/ is excluded so a deploy never deletes the trained net. The
# checkpoints_* training arms and research/ are excluded because the server has
# no use for them and they had grown to 143 MB of pointless transfer per deploy.
# Note rsync's --delete leaves excluded paths on the server alone, which is the
# point: the live net survives. Old arms already up there stay until removed by
# hand — do NOT reach for --delete-excluded, it would take checkpoints/ with it.
rsync -az --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache' \
  --exclude 'checkpoints' \
  --exclude 'checkpoints_*' \
  --exclude 'checkpoints.*' \
  --exclude '*.launch.log' \
  --exclude 'research' \
  --exclude '_diag_*' \
  --exclude 'archive' \
  --exclude 'references' \
  --exclude 'tutorials' \
  --exclude 'tests' \
  --exclude 'c_engine' \
  ./ "$HOST:$PATH_REMOTE/"

if [[ $WITH_NET -eq 1 ]]; then
  echo "==> copying $NET -> checkpoints/best.pt"
  [[ -f "$NET" ]] || { echo "no such net: $NET" >&2; exit 1; }
  ssh "$HOST" "mkdir -p $PATH_REMOTE/checkpoints"
  rsync -az "$NET" "$HOST:$PATH_REMOTE/checkpoints/best.pt"
fi

echo "==> installing deps and restarting"
ssh "$HOST" "VENV='$VENV' APP='$PATH_REMOTE' bash -s" <<'REMOTE'
set -euo pipefail

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip -q
fi

# The CPU index matters: the default wheels are the CUDA build, several GB of
# nothing useful on a droplet.
"$VENV/bin/pip" install -q -r "$APP/requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cpu

chown -R stackit:stackit /opt/stackit
systemctl restart stackit
sleep 3
systemctl is-active stackit
curl -sf localhost:3300/healthz && echo
REMOTE

echo "==> done: https://stackit.localgeek.jp"
