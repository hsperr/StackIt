# StackIt AlphaZero

A small, from-scratch AlphaZero for StackIt: a neural net that learns the game
purely by playing itself, guided by Monte-Carlo Tree Search. No human games, no
handcrafted evaluation — it bootstraps from random play.

## Quick start

```bash
# 1. Train (4x4 board by default). Writes checkpoints + live metrics.
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

## Device note

MCTS self-play/eval run on **CPU** (batch-1 forwards are faster there for this
tiny net); training runs batched on **MPS/CUDA** if available. Override the
train device with `--device cpu|mps|cuda`.

## Key knobs (in `config.py` or via CLI)

- `--sims` — MCTS simulations per move. The biggest quality/speed lever.
- `--games` — self-play games per iteration (data volume/freshness).
- `--board` — board size (start at 4, scale up once it's learning).
- `res_blocks` / `channels` — net size. Scale up with the board.
