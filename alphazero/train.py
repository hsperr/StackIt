"""The AlphaZero training loop for StackIt.

One iteration = self-play (generate data) -> train (fit net to MCTS policy +
outcomes) -> gate (candidate vs previous best; revert if it doesn't clear the
win threshold) -> benchmark (vs random & AlphaBeta for an absolute signal).

Device split: MCTS self-play/eval run on CPU (batch-1 forwards are faster there
on this tiny net); training runs batched on MPS/CUDA if available.

Run:  python3 -m alphazero.train                 # full run, sane 4x4 defaults
      python3 -m alphazero.train --quick          # tiny smoke test
      python3 -m alphazero.train --iterations 100 --sims 80
"""
import os
import copy
import time
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .net import StackNet, resolve_device
from .replay import ReplayBuffer
from .selfplay import expand_symmetries
from .arena_eval import win_rate
from .parallel import make_pool, selfplay_parallel, match_parallel
from .rating import ResultBook, compute_elo
from . import metrics


def train_net(net, opt, buffer, cfg, device, steps):
    net.train()
    tot_p = tot_v = 0.0
    for _ in range(steps):
        planes, pi, z = buffer.sample(cfg.batch_size, device)
        logits, v = net(planes)
        policy_loss = -(pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(v, z)
        loss = policy_loss + value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot_p += policy_loss.item()
        tot_v += value_loss.item()
    return tot_p / steps, tot_v / steps


def gate(net, old_state, cfg, pool, seed):
    """Candidate (net) vs previous best (old_state), played in parallel.
    Returns (win_rate, accept, (wins_cand, wins_prev, draws))."""
    wa, wb, dr = match_parallel(net, ("az", old_state), cfg, cfg.eval_games,
                                seed, pool, cfg.num_workers)
    wr = win_rate(wa, wb, dr)
    return wr, wr >= cfg.eval_win_threshold, (wa, wb, dr)


def sample_opponents(prev_ids, k, rng):
    """Pick up to k past version ids to play this iteration: always the most
    recent and the oldest, then a random spread of the rest."""
    if not prev_ids or k <= 0:
        return []
    ids = sorted(prev_ids)
    chosen = {ids[-1], ids[0]}
    rest = [i for i in ids if i not in chosen]
    rng.shuffle(rest)
    for i in rest:
        if len(chosen) >= k:
            break
        chosen.add(i)
    return sorted(chosen)[:k]


def run(cfg, resume=False):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cpu = torch.device("cpu")
    train_device = resolve_device(cfg.device)
    print(f"[cfg] board={cfg.board_size}x{cfg.board_size} sims={cfg.num_simulations} "
          f"games/iter={cfg.games_per_iter} train_device={train_device} selfplay=cpu")

    metrics.ensure_dir(cfg)
    if not resume:
        import shutil
        if os.path.exists(cfg.metrics_file):
            os.remove(cfg.metrics_file)
        vdir = os.path.join(cfg.ckpt_dir, "versions")
        if os.path.isdir(vdir):
            shutil.rmtree(vdir)             # stale version snapshots would skew Elo
        rpath = os.path.join(cfg.ckpt_dir, "ratings.json")
        if os.path.exists(rpath):
            os.remove(rpath)

    net = StackNet(cfg.board_size, cfg.channels, cfg.res_blocks)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.replay_capacity)
    pool = make_pool(cfg.num_workers)
    print(f"[cfg] {cfg.num_workers} CPU self-play/eval workers")

    rng = np.random.default_rng(cfg.seed + 777)
    book = ResultBook()                     # every match played, for Elo
    metrics.save_version(cfg, net, 0)       # v0 = the initial (random) net
    saved_ids = [0]                         # iteration ids with a saved snapshot
    best_iter = 0                           # iteration id of the current best model

    try:
      for it in range(1, cfg.iterations + 1):
        t_iter = time.time()

        # ---------------------------------------------------- self-play (CPU pool)
        net.to(cpu)
        metrics.write_status(cfg, {"iter": it, "phase": "self-play",
                                   "games": cfg.games_per_iter, "buffer": len(buffer)})
        t0 = time.time()
        examples, last_record = selfplay_parallel(
            net, cfg, cfg.games_per_iter, base_seed=cfg.seed + it * 100000,
            pool=pool, n_workers=cfg.num_workers)
        aug = expand_symmetries(examples, cfg)
        buffer.add_many(aug)
        added = len(aug)
        if last_record is not None:
            metrics.write_last_game(cfg, {**last_record, "iter": it})
        sp_time = time.time() - t0

        # ------------------------------------------------------ train (GPU)
        metrics.write_status(cfg, {"iter": it, "phase": "train", "buffer": len(buffer)})
        old_state = copy.deepcopy(net.state_dict())
        net.to(train_device)
        steps = cfg.train_steps_per_iter if len(buffer) >= cfg.batch_size else 0
        if steps:
            pol_loss, val_loss = train_net(net, opt, buffer, cfg, train_device, steps)
        else:
            pol_loss = val_loss = float("nan")

        # ------------------- rate THIS iteration's model, on one absolute scale
        # `net` currently holds candidate_it (this iteration's trained weights).
        # We rate it as its own version `vit`, gated against the current best.
        net.to(cpu)
        base = cfg.seed + it * 100000
        cand_id = f"v{it}"
        cand_wr = None
        accepted = False
        wr_random = wr_ab = 0.0
        elo = {}
        cand_elo = best_elo = 0.0
        opponent_iter = best_iter          # the current best this iteration plays

        if steps:
            metrics.save_version(cfg, net, it)     # snapshot candidate_it
            saved_ids.append(it)

            # gate: candidate_it vs current best (best_iter)
            metrics.write_status(cfg, {"iter": it, "phase": "gate", "buffer": len(buffer)})
            cand_wr, accepted, (wa, wb, dr) = gate(net, old_state, cfg, pool, base + 50000)
            book.add_match(cand_id, f"v{best_iter}", wa, wb, dr)

            # benchmark candidate vs random + alphabeta (per-row win rates)
            metrics.write_status(cfg, {"iter": it, "phase": "benchmark", "buffer": len(buffer)})
            wa, wb, dr = match_parallel(net, ("random", None), cfg, cfg.benchmark_games,
                                        base + 70000, pool, cfg.num_workers)
            wr_random = win_rate(wa, wb, dr)
            book.add_match(cand_id, "random", wa, wb, dr)
            wa, wb, dr = match_parallel(net, ("alphabeta", cfg.alphabeta_budget), cfg,
                                        cfg.benchmark_games, base + 75000, pool, cfg.num_workers)
            wr_ab = win_rate(wa, wb, dr)
            book.add_match(cand_id, "alphabeta", wa, wb, dr)

            # gauntlet: candidate vs a spread of PAST iterations (Elo connectivity)
            prev = [i for i in saved_ids if i != it and i != best_iter]
            for oid in sample_opponents(prev, cfg.gauntlet_versions, rng):
                ostate = metrics.load_version_state(cfg, oid)
                wa, wb, dr = match_parallel(net, ("az", ostate), cfg, cfg.gauntlet_games,
                                            base + 80000 + oid, pool, cfg.num_workers)
                book.add_match(cand_id, f"v{oid}", wa, wb, dr)

            elo = compute_elo(book, anchor="random")
            cand_elo = elo.get(cand_id, 0.0)

            # accept -> candidate becomes the new best; reject -> revert weights
            if accepted:
                best_iter = it
            else:
                net.load_state_dict(old_state)
            best_elo = elo.get(f"v{best_iter}", 0.0)

        metrics.write_ratings(cfg, {
            "anchor": "random", "best_iter": best_iter,
            "elo": {k: round(v, 1) for k, v in elo.items()},
            "versions": [{"iter": i, "elo": round(elo.get(f"v{i}", 0.0), 1),
                          "best": i == best_iter} for i in saved_ids],
        })

        # best.pt = current best weights (net was reverted above if rejected)
        metrics.save_checkpoint(cfg, net, "best.pt",
                                extra={"iter": best_iter, "version": best_iter,
                                       "elo": round(best_elo, 1)})
        row = {
            "iter": it,
            "version": opponent_iter,          # current best this iteration played against
            "elo": round(cand_elo, 1),         # THIS iteration's model, per row
            "best_elo": round(best_elo, 1),
            "elo_alphabeta": round(elo.get("alphabeta", 0.0), 1),
            "policy_loss": round(pol_loss, 4),
            "value_loss": round(val_loss, 4),
            "cand_winrate_vs_best": None if cand_wr is None else round(cand_wr, 3),
            "accepted": accepted,
            "winrate_vs_random": round(wr_random, 3),
            "winrate_vs_alphabeta": round(wr_ab, 3),
            "buffer": len(buffer),
            "new_examples": added,
            "selfplay_sec": round(sp_time, 1),
            "iter_sec": round(time.time() - t_iter, 1),
        }
        metrics.append_metric(cfg, row)
        print(f"[iter {it:3d}] v{it} elo={cand_elo:.0f} best=v{best_iter}({best_elo:.0f}) "
              f"loss(p/v)={pol_loss:.3f}/{val_loss:.3f} "
              f"vs_rand={wr_random:.2f} vs_AB={wr_ab:.2f} "
              f"gate={'-' if cand_wr is None else f'{cand_wr:.2f}'}"
              f"{'' if accepted else ' (rej)'} {row['iter_sec']}s")

      metrics.write_status(cfg, {"iter": cfg.iterations, "phase": "done"})
    finally:
        pool.close()
        pool.join()


# --------------------------------------------------------------------- CLI
def parse_args():
    d = Config()
    ap = argparse.ArgumentParser(description="Train StackIt AlphaZero")
    ap.add_argument("--iterations", type=int, default=d.iterations)
    ap.add_argument("--games", type=int, default=d.games_per_iter, dest="games_per_iter")
    ap.add_argument("--sims", type=int, default=d.num_simulations, dest="num_simulations")
    ap.add_argument("--board", type=int, default=d.board_size, dest="board_size")
    ap.add_argument("--train-steps", type=int, default=d.train_steps_per_iter,
                    dest="train_steps_per_iter")
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke test: few games/sims/iters")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = Config(iterations=a.iterations, games_per_iter=a.games_per_iter,
                 num_simulations=a.num_simulations, board_size=a.board_size,
                 train_steps_per_iter=a.train_steps_per_iter,
                 device=a.device, seed=a.seed)
    if a.quick:
        cfg.iterations = min(cfg.iterations, 3)
        cfg.games_per_iter = 4
        cfg.num_simulations = 25
        cfg.train_steps_per_iter = 50
        cfg.eval_games = 6
        cfg.benchmark_games = 6
    run(cfg, resume=a.resume)


if __name__ == "__main__":
    main()
