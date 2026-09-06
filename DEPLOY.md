# Deploying the StackIt server

Target: a small Linux box (1 GB RAM, 1–2 cores) with an existing reverse proxy.
The app is a single Flask process behind gunicorn. There is no database, no
build step, and no external service.

## What the server needs

| | |
|---|---|
| Python | 3.10 or newer |
| RAM | ~250 MB resident with AlphaZero loaded, ~55 MB without |
| Disk | ~600 MB, almost all of it PyTorch (~500 MB) |
| CPU | 1 core is enough. Each AI move keeps one core busy for up to 5 seconds |
| Network | one HTTP port, proxied |
| Extra files | `checkpoints/best.pt` — **not in git**, you must copy it yourself |

These are measured, not estimated: 211 MB peak resident for a process that has
loaded the net and run a search, 35 MB for one running an MCTS game.

On 1 GB this fits, but not with much to spare. Add a 1 GB swap file if the box
has none — PyTorch's import peak is the tightest moment.

## The one rule: a single worker

Games live in the process's memory, and an online game is shared by several
browsers at once, so every one of them must land in the same process.
Run more than one gunicorn **worker** and
each request lands on a random process, so half of them cannot find the game.
Verified: with `-w 3`, 6 of 12 lookups of a live game returned 404.

Use one worker and several threads:

```
gunicorn -w 1 --threads 8 ...
```

Threads are fine — an AI move does not block other players. Measured with a
6-second AlphaZero move in flight, unrelated requests still answered in 0.04 s.
On a single-core box two people moving at once will each wait longer, because
they share the one core.

## Setup

Run these on the server as root, or with `sudo`.

```bash
# 1. System packages
apt update && apt install -y python3 python3-venv python3-pip git

# 2. A user to run it as, and a place to put it
useradd --system --create-home --home-dir /opt/stackit --shell /usr/sbin/nologin stackit
cd /opt/stackit
sudo -u stackit git clone <YOUR_REPO_URL> app
cd app

# 3. Virtualenv. The --extra-index-url matters: without it pip installs the
#    CUDA build of torch and pulls in several gigabytes you have no use for.
sudo -u stackit python3 -m venv /opt/stackit/venv
sudo -u stackit /opt/stackit/venv/bin/pip install --upgrade pip
sudo -u stackit /opt/stackit/venv/bin/pip install -r requirements.txt \
     --extra-index-url https://download.pytorch.org/whl/cpu

# 4. The trained network. It is gitignored, so copy it from your machine:
#      scp checkpoints/best.pt user@server:/tmp/best.pt
mkdir -p /opt/stackit/app/checkpoints
mv /tmp/best.pt /opt/stackit/app/checkpoints/best.pt
chown -R stackit:stackit /opt/stackit
```

Without `checkpoints/best.pt` everything still works — AlphaZero simply shows
as unavailable in the menu, with the reason. The net is trained for one board
size; on any other size the option is disabled and says so.

## systemd service

Write `/etc/systemd/system/stackit.service`:

```ini
[Unit]
Description=StackIt game server
After=network.target

[Service]
Type=simple
User=stackit
WorkingDirectory=/opt/stackit/app
ExecStart=/opt/stackit/venv/bin/gunicorn \
    --workers 1 \
    --threads 8 \
    --timeout 120 \
    --bind 127.0.0.1:9999 \
    --access-logfile - \
    server:app
Restart=always
RestartSec=5

# torch spawns a thread per core for maths it does not need here, which just
# fights the web threads on a small box.
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1

# Keep a runaway process from taking the whole box down with it.
MemoryMax=700M

[Install]
WantedBy=multi-user.target
```

`--timeout 120` matters: gunicorn kills a worker that looks stuck, and an AI
move legitimately blocks its thread for several seconds.

Then:

```bash
systemctl daemon-reload
systemctl enable --now stackit
systemctl status stackit
curl -s localhost:9999/healthz          # {"games":0,"ok":true}
```

## Reverse proxy

It listens on `127.0.0.1:9999`, so put your existing proxy in front. nginx:

```nginx
location / {
    proxy_pass http://127.0.0.1:9999;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 120s;          # AI moves take seconds
}
```

Caddy:

```
yourdomain.com {
    reverse_proxy 127.0.0.1:9999
}
```

## Limits, and where to change them

A visitor chooses the board size and the thinking time, so both are capped in
code. They are constants at the top of `server.py`, deliberately not settings —
there is nothing to configure at deploy time:

| constant | value | why |
|---|---|---|
| `MAX_THINKING_TIME` | 5 s | the longest one request can hold a core |
| `MAX_BOARD` | 8 | an engine has to search it, and cost climbs steeply |
| `MAX_BOARD_HUMANS` | 12 | nothing searches a humans-only game |
| `MAX_PLAYERS` | 5 | seats in a humans-only game |
| `MAX_ENGINE_GAMES` | 16 | live "you vs computer" games; the oldest is evicted |
| `MAX_HUMAN_GAMES` | 48 | live one-screen and online games, evicted the same way |
| `MAX_EVENTS` | 4 | recent moves kept so a slow browser can catch up |
| `GAME_TTL` | 1 h | a game nobody has touched for this long is dropped |

The two game caps are separate because the two kinds cost wildly different
amounts of memory, and online play needs many cheap games at once:

| kind | measured | why |
|---|---|---|
| "you vs computer", MCTS | 12 MB after 60 plies, still growing | the search tree is kept between moves and never shrinks |
| online / one screen | 1.15 MB after 3000 plies on 12x12 with 5 seats | no search tree, and no undo stack |

An online game keeps no undo stack at all — it refuses take-backs, so there is
nothing to keep. Without that it was 13.5 MB instead of 1.15 MB.

Sixteen engine games is the worst case this box can take alongside PyTorch, so
that number has not moved. Raise it together with the RAM, not on its own.

## Online games and the proxy

Each browser in an online game polls once a second, so a five-player game is
five small requests a second. They are cheap — a poll is one `state()` build,
no search — but they are constant, so leave `--threads 8` alone and do not put
a tight connection limit in front of it.

There is no login and no rate limiting. A join code is four characters from a
32-letter alphabet, about a million combinations, which is enough to stop a
friend guessing but not a script. Nothing sensitive sits behind one — the worst
case is a stranger taking a seat in your game. If it will be publicly reachable,
rate limit `/api/join` at the proxy.

## Updating

```bash
cd /opt/stackit/app
sudo -u stackit git pull
sudo -u stackit /opt/stackit/venv/bin/pip install -r requirements.txt \
     --extra-index-url https://download.pytorch.org/whl/cpu
systemctl restart stackit
```

`checkpoints/` is gitignored, so a pull never touches the trained net. To ship a
newly trained one, copy the file over and restart — the net is loaded once at
first use and cached.

## If something is wrong

| symptom | cause |
|---|---|
| "That game expired. Start a new one." right after starting | more than one gunicorn worker |
| AlphaZero missing from the menu | no `checkpoints/best.pt`, or it is for a different board size. The menu shows the reason |
| worker killed and restarted during play | `--timeout` too low for the thinking time |
| OOM kill | `MemoryMax` too low, PyTorch installed as the CUDA build, or no swap |
| slow first AlphaZero move | the net loads lazily on first use; only the first request pays |

Logs: `journalctl -u stackit -f`
