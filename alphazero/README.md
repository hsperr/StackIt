# StackIt AlphaZero

A small, from-scratch AlphaZero for StackIt: a neural net that learns the game
purely by playing itself, guided by Monte-Carlo Tree Search. No human games, no
handcrafted evaluation — it bootstraps from random play.

## Quick start

```bash
# 1. Train (5x5 board by default). Writes checkpoints + live metrics.
python3 -m alphazero.train

#    Tiny smoke test first if you like:
python3 -m alphazero.train --quick

# 2. In another terminal, watch it learn:
python3 -m alphazero.dashboard        # open http://localhost:8123

# 3. Play the trained net in the web UI (pick "AlphaZero" in the dropdown):
python3 server.py                     # http://localhost:9999

# 4. Or pit it against the other engines in a tournament:
python3 arena.py --engines alphabeta,mcts,alphazero --size 4
```

The engine loads `checkpoints/best.pt`, so train before using it in the
server/arena. It is trained for one board size — train a fresh net if you change
`--board`.

## How it works (the loop)

Each **iteration** is one round of policy iteration:

1. **Self-play** — play `--games` games. Every move runs `--sims` MCTS
   simulations guided by the net; the move is sampled from the search's visit
   counts. Each position stores `(state, π, player)` where `π` = visit-count
   distribution (the "improved policy").
2. **Store** — outcomes `z` are filled in (+1 win / −1 loss / 0 draw from that
   position's player's view) and pushed to a replay buffer, 8× augmented via
   board symmetries.
3. **Train** — SGD on `loss = policyCrossEntropy(π) + valueMSE(z)` + weight decay.
4. **Gate** — the updated net plays the previous best; it's kept only if it wins
   ≥ 55%, otherwise reverted (guards against a bad update poisoning self-play).
5. **Benchmark** — play vs a random player and vs the existing AlphaBeta engine
   for an absolute progress signal. All logged to `metrics.jsonl`.

## Files

| file | role |
|---|---|
| `config.py` | all hyperparameters (`Config` dataclass) |
| `encoding.py` | Board → 9 canonical planes, action indexing, symmetry augmentation |
| `net.py` | residual net (policy + value heads) + `Evaluator` for masked inference |
| `mcts_az.py` | PUCT MCTS that uses the net instead of rollouts |
| `selfplay.py` | generate games → training examples |
| `replay.py` | bounded replay buffer |
| `arena_eval.py` | in-process match play (net vs net / random / AlphaBeta) |
| `train.py` | the training loop (CLI) |
| `engine.py` | `AlphaZero` engine implementing `get_best_move(...)` for server/arena |
| `dashboard.py` | live training dashboard (Flask) |
| `metrics.py` | checkpoint + metrics/JSON I/O |
| `parallel.py` | multi-process self-play / arena workers |
| `rating.py` | Elo pool over all past versions + reference opponents |
| `ab_bench.py` | out-of-process "strong AlphaBeta" benchmark (subprocess, non-blocking) |

## Device note

MCTS self-play/eval run on **CPU** (batch-1 forwards are faster there for this
tiny net); training runs batched on **MPS/CUDA** if available. Override the
train device with `--device cpu|mps|cuda`.

## Key knobs (in `config.py` or via CLI)

- `--sims` — MCTS simulations per move. The biggest quality/speed lever.
- `--games` — self-play games per iteration (data volume/freshness).
- `--board` — board size (start at 4, scale up once it's learning).
- `--workers N` — parallel self-play/arena processes.
- `--ckpt-dir DIR` — write checkpoints/metrics here instead of `checkpoints/`.
  Use one directory per experiment so runs never mix.
- `--resume` — continue an interrupted run from `<ckpt-dir>/best.pt`.
- `res_blocks` / `channels` — net size. Scale up with the board.

## Strength upgrades (on by default)

Four research-backed additions sit on top of vanilla AlphaZero. Each has an
off-switch so you can ablate it:

| feature | what it does | disable with |
|---|---|---|
| Gumbel root MCTS | Gumbel-top-k + Sequential Halving at the root; trains on the "completed policy" instead of raw visit counts (Danihelka et al., ICLR 2022). Replaces Dirichlet noise. | `--no-gumbel` |
| Playout Cap Randomization | only ~25% of moves get a full search and are recorded; the rest run cheap and unrecorded. | `--pcr-prob 0` |
| Ownership head | per-cell 3-way (empty/mine/theirs) auxiliary target, KataGo-style. | `--own-weight 0` |
| Tree reuse | carry the chosen child's subtree to the next ply (self-play only). | `--no-tree-reuse` |

The ownership head changes the **network architecture**. Checkpoints from before
it was added cannot be resumed — archive the old `checkpoints/` and start fresh.

## Running an experiment

```bash
# One directory per experiment. Never share a --ckpt-dir between runs.
python3 -m alphazero.train --board 4 --sims 96 --workers 6 \
        --ckpt-dir _diag_4x4/conv/my_run --iterations 150

# Watch it (point the dashboard at the same directory):
python3 -m alphazero.dashboard --ckpt-dir _diag_4x4/conv/my_run

# Resume after a stop:
python3 -m alphazero.train --resume --ckpt-dir _diag_4x4/conv/my_run \
        --board 4 --sims 96
```

`_diag_*/` and all `*.pt` files are gitignored — training artifacts stay local.

## Measuring strength honestly

Self-play Elo is **circular**: it only compares the net to its own ancestors, so
it can look flat while the net improves, or climb while it does not. For a real
answer, play a fixed external opponent:

- Use a **fixed** reference (an archived champion, or AlphaBeta at a fixed
  thinking time). If you change the AlphaBeta engine, old numbers are not
  comparable — a faster AlphaBeta searches deeper and is a harder yardstick.
- Play **enough games** (16+) with `explore_plies` > 0. Deterministic 2-game
  probes quantize to {0, 0.5, 1.0} and tell you nothing.
- Judge only after **many** iterations. A 4x4 net needed ~144 iterations to
  reach full strength, with flat stretches on the way.
