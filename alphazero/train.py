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
from .parallel import make_pool, selfplay_parallel, match_parallel, net_arch_state
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


def _load_ckpt(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["arch"], d["state"]


def _az_opp(registry, pid):
    arch, state = _load_ckpt(registry[pid]["path"])
    return ("az", arch, state)


def sample_ids(ids, k, rng):
    """Pick up to k ids to play this iteration: always the most recent and the
    oldest, then a random spread of the rest (spans the strength range)."""
    if not ids or k <= 0:
        return []
    ids = sorted(ids, key=str)
    chosen = {ids[-1], ids[0]}
    rest = [i for i in ids if i not in chosen]
    rng.shuffle(rest)
    for i in rest:
        if len(chosen) >= k:
            break
        chosen.add(i)
    return list(chosen)[:k]


def seed_reference_book(book, registry, ref_ids, cfg, pool, rng):
    """Give references (and the anchors) an initial Elo by playing them vs
    AlphaBeta, vs random, and against each other before training starts."""
    for rid in ref_ids:
        arch, state = _load_ckpt(registry[rid]["path"])
        wa, wb, dr = match_parallel(arch, state, ("random", None), cfg,
                                    cfg.benchmark_games, 111, pool, cfg.num_workers)
        book.add_match(rid, "random", wa, wb, dr)
        wa, wb, dr = match_parallel(arch, state, ("alphabeta", cfg.alphabeta_budget), cfg,
                                    cfg.benchmark_games, 222, pool, cfg.num_workers)
        book.add_match(rid, "alphabeta", wa, wb, dr)
    for i in range(len(ref_ids)):
        for j in range(i + 1, len(ref_ids)):
            a_arch, a_state = _load_ckpt(registry[ref_ids[i]]["path"])
            wa, wb, dr = match_parallel(a_arch, a_state, _az_opp(registry, ref_ids[j]), cfg,
                                        cfg.gauntlet_games * 2, 333, pool, cfg.num_workers)
            book.add_match(ref_ids[i], ref_ids[j], wa, wb, dr)


def run(cfg, resume=False, references_dir=None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cpu = torch.device("cpu")
    train_device = resolve_device(cfg.device)
    print(f"[cfg] board={cfg.board_size}x{cfg.board_size} net={cfg.res_blocks}x{cfg.channels} "
          f"sims={cfg.num_simulations} games/iter={cfg.games_per_iter} "
          f"train_device={train_device} selfplay=cpu")

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

    # participant registry: id -> weights on disk (versions + references)
    registry = {}
    metrics.save_version(cfg, net, 0)       # v0 = the initial (random) net
    registry["v0"] = {"path": os.path.join(cfg.ckpt_dir, "versions", "v0.pt"),
                      "arch": net.arch(), "iter": 0, "ref": False}
    version_ids = ["v0"]

    # carry forward previous best models as fixed reference OPPONENTS (tracked in
    # the win-rate chart + Elo, like AlphaBeta — they never enter self-play or gating)
    ref_ids = []
    if references_dir and os.path.isdir(references_dir):
        for fn in sorted(os.listdir(references_dir)):
            if not fn.endswith(".pt"):
                continue
            rid = "ref:" + os.path.splitext(fn)[0]
            path = os.path.join(references_dir, fn)
            arch, _ = _load_ckpt(path)
            registry[rid] = {"path": path, "arch": arch, "iter": None, "ref": True}
            ref_ids.append(rid)
        print(f"[cfg] {len(ref_ids)} reference opponent(s): {ref_ids}")
        seed_reference_book(book, registry, ref_ids, cfg, pool, rng)

    elo = (compute_elo(book, anchor="alphabeta", anchor_elo=0.0,
                       prior_draws=cfg.elo_prior_draws) if book.d else {})
    # the reference tracked in the win-rate chart (strongest one; one is enough)
    primary_ref = max(ref_ids, key=lambda i: elo.get(i, -1e9)) if ref_ids else None
    best_iter = 0                            # accepted-lineage best (normal AlphaZero gating)

    try:
      for it in range(1, cfg.iterations + 1):
        t_iter = time.time()
        cand_id = f"v{it}"

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
        net.to(train_device)
        steps = cfg.train_steps_per_iter if len(buffer) >= cfg.batch_size else 0
        if steps:
            pol_loss, val_loss = train_net(net, opt, buffer, cfg, train_device, steps)
        else:
            pol_loss = val_loss = float("nan")

        # ---------------- gate (normal AlphaZero) + benchmarks incl. reference
        net.to(cpu)
        base = cfg.seed + it * 100000
        cand_wr = None
        accepted = False
        wr_random = wr_ab = wr_ref = 0.0
        cand_elo = 0.0
        best_elo = elo.get(f"v{best_iter}", 0.0)
        opponent_iter = best_iter          # previous best this candidate is gated against

        if steps:
            metrics.save_version(cfg, net, it)
            registry[cand_id] = {"path": os.path.join(cfg.ckpt_dir, "versions", f"v{it}.pt"),
                                 "arch": net.arch(), "iter": it, "ref": False}
            version_ids.append(cand_id)
            old_state = copy.deepcopy(net.state_dict())
            a_arch, a_state = net_arch_state(net)

            # gate: candidate vs the PREVIOUS BEST (own lineage) — decides acceptance
            metrics.write_status(cfg, {"iter": it, "phase": "gate", "buffer": len(buffer)})
            wa, wb, dr = match_parallel(a_arch, a_state, _az_opp(registry, f"v{best_iter}"), cfg,
                                        cfg.eval_games, base + 50000, pool, cfg.num_workers)
            cand_wr = win_rate(wa, wb, dr)
            accepted = cand_wr >= cfg.eval_win_threshold
            book.add_match(cand_id, f"v{best_iter}", wa, wb, dr)

            # benchmarks: random, AlphaBeta, and the reference — all just tracked opponents
            metrics.write_status(cfg, {"iter": it, "phase": "benchmark", "buffer": len(buffer)})
            wa, wb, dr = match_parallel(a_arch, a_state, ("random", None), cfg,
                                        cfg.benchmark_games, base + 70000, pool, cfg.num_workers)
            wr_random = win_rate(wa, wb, dr)
            book.add_match(cand_id, "random", wa, wb, dr)
            wa, wb, dr = match_parallel(a_arch, a_state, ("alphabeta", cfg.alphabeta_budget), cfg,
                                        cfg.benchmark_games, base + 75000, pool, cfg.num_workers)
            wr_ab = win_rate(wa, wb, dr)
            book.add_match(cand_id, "alphabeta", wa, wb, dr)
            if primary_ref is not None:
                wa, wb, dr = match_parallel(a_arch, a_state, _az_opp(registry, primary_ref), cfg,
                                            cfg.benchmark_games, base + 77000, pool, cfg.num_workers)
                wr_ref = win_rate(wa, wb, dr)
                book.add_match(cand_id, primary_ref, wa, wb, dr)

            # gauntlet: candidate vs a spread of past versions (Elo connectivity)
            others = [i for i in version_ids if i not in (cand_id, f"v{best_iter}")]
            for k, oid in enumerate(sample_ids(others, cfg.gauntlet_versions, rng)):
                wa, wb, dr = match_parallel(a_arch, a_state, _az_opp(registry, oid), cfg,
                                            cfg.gauntlet_games, base + 80000 + k * 1000,
                                            pool, cfg.num_workers)
                book.add_match(cand_id, oid, wa, wb, dr)

            elo = compute_elo(book, anchor="alphabeta", anchor_elo=0.0,
                              prior_draws=cfg.elo_prior_draws)
            cand_elo = elo.get(cand_id, 0.0)

            if accepted:
                best_iter = it
            else:
                net.load_state_dict(old_state)   # reject: keep the previous best
            best_elo = elo.get(f"v{best_iter}", 0.0)

        # ratings.json: live Elo + game counts for every participant
        games = book.games_played()
        metrics.write_ratings(cfg, {
            "anchor": "alphabeta", "best_iter": best_iter, "primary_ref": primary_ref,
            "elo": {k: round(v, 1) for k, v in elo.items()},
            "versions": [{"id": i, "iter": registry[i]["iter"], "ref": registry[i]["ref"],
                          "elo": round(elo.get(i, 0.0), 1), "games": int(games.get(i, 0)),
                          "best": i == f"v{best_iter}"} for i in registry],
            "results": book.to_list(),
        })

        # best.pt = the accepted-lineage best (net was reverted above if rejected)
        metrics.save_checkpoint(cfg, net, "best.pt",
                                extra={"iter": best_iter, "version": best_iter,
                                       "elo": round(best_elo, 1)})
        row = {
            "iter": it,
            "version": opponent_iter,           # previous best this iteration played (gate)
            "elo": round(cand_elo, 1),          # this iteration's model (AlphaBeta = 0)
            "best_elo": round(best_elo, 1),
            "elo_alphabeta": 0.0,               # AlphaBeta is the anchor now
            "policy_loss": round(pol_loss, 4),
            "value_loss": round(val_loss, 4),
            "cand_winrate_vs_best": None if cand_wr is None else round(cand_wr, 3),
            "accepted": accepted,
            "winrate_vs_random": round(wr_random, 3),
            "winrate_vs_alphabeta": round(wr_ab, 3),
            "winrate_vs_reference": round(wr_ref, 3),
            "reference_id": None if primary_ref is None else primary_ref.replace("ref:", ""),
            "buffer": len(buffer),
            "new_examples": added,
            "selfplay_sec": round(sp_time, 1),
            "iter_sec": round(time.time() - t_iter, 1),
        }
        metrics.append_metric(cfg, row)
        print(f"[iter {it:3d}] v{it} elo={cand_elo:.0f} best=v{best_iter}({best_elo:.0f}) "
              f"loss(p/v)={pol_loss:.3f}/{val_loss:.3f} vs_rand={wr_random:.2f} "
              f"vs_AB={wr_ab:.2f} vs_ref={wr_ref:.2f} "
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
    ap.add_argument("--references-dir", default=None,
                    help="dir of previous best .pt models to keep as fixed opponents")
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
    run(cfg, resume=a.resume, references_dir=a.references_dir)


if __name__ == "__main__":
    main()
