"""The AlphaZero training loop for StackIt.

One iteration = self-play (generate data) -> train (fit net to MCTS policy +
outcomes) -> gate (candidate vs previous best; revert if it doesn't clear the
win threshold) -> benchmark (vs random & AlphaBeta for an absolute signal).

Device split: MCTS self-play/eval run on CPU (batch-1 forwards are faster there
on this tiny net); training runs batched on MPS/CUDA if available.

Run:  python3 -m alphazero.train                 # full run, sane 5x5 defaults
      python3 -m alphazero.train --quick          # tiny smoke test
      python3 -m alphazero.train --iterations 100 --sims 80
"""
import os
import sys
import copy
import time
import json
import argparse
import subprocess
from dataclasses import replace

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
from .ab_bench import benchmark_running
from . import metrics


def train_net(net, opt, buffer, cfg, device, steps):
    net.train()
    nb = net.board_size
    tot_p = tot_v = tot_o = 0.0
    for _ in range(steps):
        planes, pi, z, own = buffer.sample(cfg.batch_size, device)
        logits, v, own_logits = net(planes)
        policy_loss = -(pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(v, z)
        # ownership: per-cell 3-way cross entropy (own_logits [B,3,N,N], own [B,N,N])
        own_loss = F.cross_entropy(own_logits, own.view(-1, nb, nb))
        loss = policy_loss + value_loss + cfg.own_loss_weight * own_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot_p += policy_loss.item()
        tot_v += value_loss.item()
        tot_o += own_loss.item()
    return tot_p / steps, tot_v / steps, tot_o / steps


def _load_ckpt(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["arch"], d["state"]


def _az_opp(registry, pid):
    arch, state = _load_ckpt(registry[pid]["path"])
    return ("az", arch, state)


def _launch_ab_bench(cfg, weights_path, champion_iter, train_iter):
    """Fire-and-forget the strong-AlphaBeta benchmark as a detached subprocess.
    Never blocks the training loop; skips only if a LIVE benchmark is still up
    (a stale lock from a crashed benchmark is cleared, not honored forever)."""
    if benchmark_running(cfg):
        print(f"[ab-bench] skip v{champion_iter}: previous benchmark still running")
        return
    try:
        subprocess.Popen(
            [sys.executable, "-m", "alphazero.ab_bench",
             "--weights", weights_path,
             "--champion-iter", str(champion_iter),
             "--train-iter", str(train_iter),
             "--cfg", json.dumps(cfg.to_dict())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        print(f"[ab-bench] launched for champion v{champion_iter} "
              f"({cfg.ab_bench_games} games @ {cfg.ab_bench_think}s/move)")
    except Exception as e:                        # never let a benchmark kill training
        print(f"[ab-bench] launch failed: {e}")


def _latest_ab_winrate(cfg):
    """Most recent completed strong-AB benchmark winrate, for the dashboard's
    'vs AlphaBeta' line (carried forward between benchmarks). None if none yet."""
    try:
        last = None
        with open(cfg.ab_bench_file) as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last).get("winrate") if last else None
    except (FileNotFoundError, ValueError):
        return None


def _version_num(vid):
    """Numeric iteration of a 'v<N>' id (e.g. v71 -> 71), for correct ordering.
    Lexicographic sorting would rank v9 above v71."""
    tail = vid.rsplit(":", 1)[-1]           # tolerate a 'ref:' prefix
    return int(tail[1:]) if tail[1:].isdigit() else -1


def sample_ids(ids, k, rng):
    """Pick up to k ids to play this iteration: always the most recent and the
    oldest, then a random spread of the rest (spans the strength range)."""
    if not ids or k <= 0:
        return []
    ids = sorted(ids, key=_version_num)
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

    net = StackNet(cfg.board_size, cfg.channels, cfg.res_blocks)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.replay_capacity)
    pool = make_pool(cfg.num_workers)
    print(f"[cfg] {cfg.num_workers} CPU self-play/eval workers")

    rng = np.random.default_rng(cfg.seed + 777)
    book = ResultBook()                     # every match played, for Elo
    registry = {}                           # participant id -> weights on disk
    version_ids = []
    start_iter = 0                          # resume: first NEW iteration to run
    best_iter = 0
    n_champions = 0
    champion_ids = []                       # accepted-lineage ids, for the self-play pool

    # --resume: load best.pt weights + restore version registry, match history, and
    # the iteration counter, so training genuinely continues (not just "don't wipe").
    resumed = False
    if resume:
        best_path = os.path.join(cfg.ckpt_dir, "best.pt")
        ratings_path = os.path.join(cfg.ckpt_dir, "ratings.json")
        vdir = os.path.join(cfg.ckpt_dir, "versions")
        if os.path.exists(best_path) and os.path.exists(ratings_path) and os.path.isdir(vdir):
            net.load_state_dict(torch.load(best_path, map_location="cpu",
                                           weights_only=False)["state"])
            with open(ratings_path) as f:
                rd = json.load(f)
            book = ResultBook.from_list(rd.get("results", []))
            best_iter = rd.get("best_iter", 0)
            for fn in sorted(os.listdir(vdir)):
                if not fn.endswith(".pt"):
                    continue
                vid = os.path.splitext(fn)[0]
                arch, _ = _load_ckpt(os.path.join(vdir, fn))
                iter_num = int(vid[1:]) if vid[1:].isdigit() else None
                registry[vid] = {"path": os.path.join(vdir, fn), "arch": arch,
                                 "iter": iter_num, "ref": False}
                version_ids.append(vid)
            accepted_iters = []
            if os.path.exists(cfg.metrics_file):
                with open(cfg.metrics_file) as f:
                    rows = [json.loads(l) for l in f if l.strip()]
                if rows:
                    start_iter = max(r["iter"] for r in rows)
                    accepted_iters = [r["iter"] for r in rows if r.get("accepted")]
                    n_champions = len(accepted_iters)
            # rebuild the full champion lineage (not just the current best) so the
            # self-play pool survives a restart.
            champion_ids = ["v0"] + [f"v{i}" for i in accepted_iters
                                     if f"v{i}" in registry]
            resumed = True
            print(f"[resume] best=v{best_iter}, {len(version_ids)} versions, "
                  f"{len(book.d)} match records, {n_champions} champions -> "
                  f"continuing from iter {start_iter + 1}")
        else:
            print("[resume] no prior run found in checkpoints/ — starting fresh")

    if not resumed:
        import shutil
        if os.path.exists(cfg.metrics_file):
            os.remove(cfg.metrics_file)
        vdir = os.path.join(cfg.ckpt_dir, "versions")
        if os.path.isdir(vdir):
            shutil.rmtree(vdir)             # stale version snapshots would skew Elo
        rpath = os.path.join(cfg.ckpt_dir, "ratings.json")
        if os.path.exists(rpath):
            os.remove(rpath)
        # drop the previous run's strong-AB benchmark rows (+ any lock): otherwise
        # the dashboard plots stale AB winrates and iter-1 reads a bogus baseline.
        # But NOT while an old benchmark is still alive — it would append its result
        # into the freshly cleared file. Abort so the user deals with it first.
        if benchmark_running(cfg):
            raise SystemExit(
                "[fresh] a strong-AlphaBeta benchmark from a previous run is still "
                "running. Wait for it to finish or kill it before starting a fresh "
                "run (its result would contaminate the new benchmark file).")
        for stale in (cfg.ab_bench_file, cfg.ab_bench_file + ".lock"):
            if os.path.exists(stale):
                os.remove(stale)
        metrics.save_version(cfg, net, 0)   # v0 = the initial (random) net
        registry["v0"] = {"path": os.path.join(cfg.ckpt_dir, "versions", "v0.pt"),
                          "arch": net.arch(), "iter": 0, "ref": False}
        version_ids = ["v0"]
        champion_ids = ["v0"]               # the initial (random) net is the first "best"

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

    elo = (compute_elo(book, anchor="random", anchor_elo=0.0,
                       prior_draws=cfg.elo_prior_draws) if book.d else {})
    # the reference tracked in the win-rate chart (strongest one; one is enough)
    primary_ref = max(ref_ids, key=lambda i: elo.get(i, -1e9)) if ref_ids else None
    # gate/gauntlet/benchmark matches run at reduced sims — self-play keeps full sims
    ecfg = replace(cfg, num_simulations=cfg.eval_simulations)

    try:
      for it in range(start_iter + 1, cfg.iterations + 1):
        t_iter = time.time()
        cand_id = f"v{it}"

        # ---------------------------------------------------- self-play (CPU pool)
        net.to(cpu)
        metrics.write_status(cfg, {"iter": it, "phase": "self-play",
                                   "games": cfg.games_per_iter, "buffer": len(buffer)})
        t0 = time.time()
        # mix a fraction (cfg.pool_play_frac) of games vs recent PAST champions.
        # Exclude champion_ids[-1]: it's the current best == the net generating
        # self-play, so pooling it just plays identical weights (no diversity, no
        # tree reuse, opponent turns searched at full cost but unrecorded).
        past_champions = champion_ids[:-1]
        pool_ids = past_champions[-cfg.pool_size:] if cfg.pool_size > 0 else []
        opponent_pool = [_load_ckpt(registry[i]["path"]) for i in pool_ids if i in registry]
        examples, last_record = selfplay_parallel(
            net, cfg, cfg.games_per_iter, base_seed=cfg.seed + it * 100000,
            pool=pool, n_workers=cfg.num_workers, opponent_pool=opponent_pool)
        aug = expand_symmetries(examples, cfg)
        buffer.add_many(aug)
        added = len(aug)
        if last_record is not None:
            metrics.write_last_game(cfg, {**last_record, "iter": it})
        sp_time = time.time() - t0

        # ------------------------------------------------------ train (GPU)
        metrics.write_status(cfg, {"iter": it, "phase": "train", "buffer": len(buffer)})
        net.to(train_device)
        # Snapshot the PREVIOUS best (weights + optimizer momentum) BEFORE training,
        # so a rejected candidate can be fully reverted — net AND optimizer state.
        # (Snapshotting after training would capture the candidate, not the best.)
        prev_best_state = copy.deepcopy(net.state_dict())
        prev_best_opt = copy.deepcopy(opt.state_dict())
        steps = cfg.train_steps_per_iter if len(buffer) >= cfg.batch_size else 0
        if steps:
            pol_loss, val_loss, own_loss = train_net(net, opt, buffer, cfg, train_device, steps)
        else:
            pol_loss = val_loss = own_loss = float("nan")

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
            a_arch, a_state = net_arch_state(net)

            # gate: candidate vs the PREVIOUS BEST (own lineage) — decides acceptance.
            # eval matches run at reduced sims (ecfg) — same for both sides, so fair.
            metrics.write_status(cfg, {"iter": it, "phase": "gate", "buffer": len(buffer)})
            wa, wb, dr = match_parallel(a_arch, a_state, _az_opp(registry, f"v{best_iter}"), ecfg,
                                        cfg.eval_games, base + 50000, pool, cfg.num_workers)
            cand_wr = win_rate(wa, wb, dr)
            accepted = cand_wr >= cfg.eval_win_threshold
            book.add_match(cand_id, f"v{best_iter}", wa, wb, dr)

            # benchmarks: random (Elo anchor) + reference. AlphaBeta is NOT here —
            # it runs off the hot loop as a strong background benchmark (ab_bench.py).
            metrics.write_status(cfg, {"iter": it, "phase": "benchmark", "buffer": len(buffer)})
            wa, wb, dr = match_parallel(a_arch, a_state, ("random", None), ecfg,
                                        cfg.benchmark_games, base + 70000, pool, cfg.num_workers)
            wr_random = win_rate(wa, wb, dr)
            book.add_match(cand_id, "random", wa, wb, dr)

            # fixed-budget AlphaBeta: a stable sparring partner rated in the SAME
            # Elo pool. Its rating firms up as champions keep playing it, and you
            # can watch the champions climb past it over time. (This is the light
            # 0.3s AB; the heavier 5s benchmark still runs off-loop in ab_bench.)
            if cfg.ab_elo_games > 0:
                wa, wb, dr = match_parallel(a_arch, a_state, ("alphabeta", cfg.alphabeta_budget),
                                            ecfg, cfg.ab_elo_games, base + 73000, pool, cfg.num_workers)
                book.add_match(cand_id, "alphabeta", wa, wb, dr)

            if primary_ref is not None:
                wa, wb, dr = match_parallel(a_arch, a_state, _az_opp(registry, primary_ref), ecfg,
                                            cfg.benchmark_games, base + 77000, pool, cfg.num_workers)
                wr_ref = win_rate(wa, wb, dr)
                book.add_match(cand_id, primary_ref, wa, wb, dr)

            # gauntlet: candidate vs a spread of past versions (Elo connectivity)
            others = [i for i in version_ids if i not in (cand_id, f"v{best_iter}")]
            for k, oid in enumerate(sample_ids(others, cfg.gauntlet_versions, rng)):
                wa, wb, dr = match_parallel(a_arch, a_state, _az_opp(registry, oid), ecfg,
                                            cfg.gauntlet_games, base + 80000 + k * 1000,
                                            pool, cfg.num_workers)
                book.add_match(cand_id, oid, wa, wb, dr)

            elo = compute_elo(book, anchor="random", anchor_elo=0.0,
                              prior_draws=cfg.elo_prior_draws)
            cand_elo = elo.get(cand_id, 0.0)

            if accepted:
                best_iter = it
                champion_ids.append(cand_id)
                n_champions += 1
                # every Nth champion, kick off the decoupled strong-AB benchmark
                if n_champions % cfg.ab_bench_every == 0:
                    _launch_ab_bench(cfg, registry[cand_id]["path"], it, it)
            else:
                # reject: fully restore the previous best — weights AND optimizer
                # momentum — so the rejected candidate does not leak forward. Restore
                # WHILE on train_device: load_state_dict casts optimizer tensors to
                # the parameters' current device, so restoring on CPU would strand
                # Adam's momentum on CPU and crash the next opt.step() on MPS/CUDA.
                net.to(train_device)
                net.load_state_dict(prev_best_state)
                opt.load_state_dict(prev_best_opt)
                net.to(cpu)                       # back to CPU for the rest of the loop
            best_elo = elo.get(f"v{best_iter}", 0.0)

        # ratings.json: live Elo + game counts for every participant
        games = book.games_played()
        version_rows = [{"id": i, "iter": registry[i]["iter"], "ref": registry[i]["ref"],
                         "elo": round(elo.get(i, 0.0), 1), "games": int(games.get(i, 0)),
                         "best": i == f"v{best_iter}"} for i in registry]
        # AlphaBeta rides in the same ranking as a fixed opponent-to-beat (a bar on
        # the Elo chart) once it has played any games.
        if "alphabeta" in elo:
            version_rows.append({"id": "alphabeta", "iter": None, "ref": True,
                                 "elo": round(elo["alphabeta"], 1),
                                 "games": int(games.get("alphabeta", 0)), "best": False})
        metrics.write_ratings(cfg, {
            "anchor": "random", "best_iter": best_iter, "primary_ref": primary_ref,
            "elo": {k: round(v, 1) for k, v in elo.items()},
            "versions": version_rows,
            "results": book.to_list(),
        })

        # best.pt = the accepted-lineage best (net was reverted above if rejected)
        metrics.save_checkpoint(cfg, net, "best.pt",
                                extra={"iter": best_iter, "version": best_iter,
                                       "elo": round(best_elo, 1)})
        # dashboard 'vs AlphaBeta' line = latest strong-AB benchmark (carried forward)
        ab_wr = _latest_ab_winrate(cfg)
        row = {
            "iter": it,
            "version": opponent_iter,           # previous best this iteration played (gate)
            "elo": round(cand_elo, 1),          # this iteration's model (random = 0)
            "best_elo": round(best_elo, 1),
            "elo_alphabeta": 0.0,               # random is the anchor now
            "policy_loss": round(pol_loss, 4),
            "value_loss": round(val_loss, 4),
            "own_loss": round(own_loss, 4),
            "cand_winrate_vs_best": None if cand_wr is None else round(cand_wr, 3),
            "accepted": accepted,
            "winrate_vs_random": round(wr_random, 3),
            "winrate_vs_alphabeta": None if ab_wr is None else round(ab_wr, 3),
            "winrate_vs_reference": round(wr_ref, 3),
            "reference_id": None if primary_ref is None else primary_ref.replace("ref:", ""),
            "buffer": len(buffer),
            "new_examples": added,
            "selfplay_sec": round(sp_time, 1),
            "iter_sec": round(time.time() - t_iter, 1),
        }
        metrics.append_metric(cfg, row)
        print(f"[iter {it:3d}] v{it} elo={cand_elo:.0f} best=v{best_iter}({best_elo:.0f}) "
              f"loss(p/v/o)={pol_loss:.3f}/{val_loss:.3f}/{own_loss:.3f} vs_rand={wr_random:.2f} "
              f"vs_AB(5s)={'-' if ab_wr is None else f'{ab_wr:.2f}'} "
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
    # --- ablation / experiment knobs ---
    ap.add_argument("--ckpt-dir", default=None, dest="ckpt_dir",
                    help="output dir (overrides checkpoints/); metrics+ab_bench follow it")
    ap.add_argument("--workers", type=int, default=None, dest="num_workers")
    ap.add_argument("--no-gumbel", action="store_true",
                    help="ABLATION: classic PUCT visit-count target instead of Gumbel")
    ap.add_argument("--pcr-prob", type=float, default=None, dest="pcr_prob",
                    help="ABLATION: fraction of moves recorded (1.0 = record every move)")
    ap.add_argument("--own-weight", type=float, default=None, dest="own_loss_weight",
                    help="ABLATION: ownership-head loss weight (0.0 disables it)")
    ap.add_argument("--no-tree-reuse", action="store_true",
                    help="ABLATION: disable subtree carry-over in self-play")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = Config(iterations=a.iterations, games_per_iter=a.games_per_iter,
                 num_simulations=a.num_simulations, board_size=a.board_size,
                 train_steps_per_iter=a.train_steps_per_iter,
                 device=a.device, seed=a.seed)
    # ablation / experiment overrides
    if a.ckpt_dir is not None:
        cfg.ckpt_dir = a.ckpt_dir
        cfg.metrics_file = os.path.join(a.ckpt_dir, "metrics.jsonl")
        cfg.ab_bench_file = os.path.join(a.ckpt_dir, "ab_bench.jsonl")
    if a.num_workers is not None:
        cfg.num_workers = a.num_workers
    if a.no_gumbel:
        cfg.self_play_gumbel = False
    if a.pcr_prob is not None:
        cfg.pcr_prob = a.pcr_prob
    if a.own_loss_weight is not None:
        cfg.own_loss_weight = a.own_loss_weight
    if a.no_tree_reuse:
        cfg.tree_reuse = False
    if a.quick:
        cfg.iterations = min(cfg.iterations, 3)
        cfg.games_per_iter = 4
        cfg.num_simulations = 25
        cfg.eval_simulations = 16
        cfg.train_steps_per_iter = 50
        cfg.eval_games = 6
        cfg.benchmark_games = 6
        cfg.ab_elo_games = 2
        cfg.ab_bench_every = 1       # exercise the background benchmark
        cfg.ab_bench_think = 0.15
        cfg.ab_bench_games = 2
        cfg.ab_bench_workers = 2
    run(cfg, resume=a.resume, references_dir=a.references_dir)


if __name__ == "__main__":
    main()
