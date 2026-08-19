"""The AlphaZero training loop for StackIt.

One iteration = self-play (generate data) -> train (fit net to MCTS policy +
outcomes) -> evaluate (vs previous best, past versions, AlphaBeta, random) for
Elo and a progress signal.

Gating is OFF by default (cfg.use_gate) — modern AlphaZero and KataGo do not
gate. Set it True for the classic "revert the candidate unless it beats the
previous best" behaviour.

Device split: MCTS self-play/eval run on CPU (batch-1 forwards are faster there
on this tiny net); training runs batched on MPS/CUDA if available.

Run:  python3 -m alphazero.train                 # full run, sane 5x5 defaults
      python3 -m alphazero.train --quick          # tiny smoke test
      python3 -m alphazero.train --iterations 100 --sims 80
"""
import os
import sys
import copy
import math
import time
import json
import argparse
import subprocess
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .net import StackNet, build_net, resolve_device
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
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        tot_p += policy_loss.item()
        tot_v += value_loss.item()
        tot_o += own_loss.item()
    return tot_p / steps, tot_v / steps, tot_o / steps


@torch.no_grad()
def eval_net(net, val_buffer, cfg, device):
    """Held-out metrics. These positions were split off before augmentation and
    never entered the replay buffer, so this is the only number here that can
    show overfitting: watch the gap between train and val policy loss.

    Also returns two accuracies that read more directly than a loss:
      * top1  — how often the net's best move IS the search's best move
                (starts at ~1/actions, climbs as the policy learns)
      * vsign — how often the value head gets the winner's sign right
                (starts at 0.5)"""
    n = len(val_buffer)
    if n == 0:
        return (float("nan"),) * 4
    net.eval()
    nb = net.board_size
    tp = tv = 0.0
    hit = sign_hit = sign_tot = 0
    seen = 0
    for i in range(0, n, cfg.batch_size):
        chunk = [val_buffer.buf[j] for j in range(i, min(i + cfg.batch_size, n))]
        planes = torch.from_numpy(np.stack([c[0] for c in chunk])).to(device)
        pi = torch.from_numpy(np.stack([c[1] for c in chunk])).to(device)
        # c[4] is the raw game result when cfg.value_q_ratio blends the training
        # target; score held-out value against THAT so the number means the same
        # thing across runs. Falls back to c[2] for data recorded without it.
        z = torch.from_numpy(np.array([(c[4] if len(c) > 4 else c[2]) for c in chunk],
                                      dtype=np.float32)).to(device)
        logits, v, _ = net(planes)
        b = len(chunk)
        tp += float(-(pi * F.log_softmax(logits, dim=1)).sum(dim=1).sum())
        tv += float(F.mse_loss(v, z, reduction="sum"))
        hit += int((logits.argmax(dim=1) == pi.argmax(dim=1)).sum())
        decided = z != 0
        sign_tot += int(decided.sum())
        sign_hit += int(((torch.sign(v) == torch.sign(z)) & decided).sum())
        seen += b
    return (tp / seen, tv / seen, hit / seen,
            sign_hit / sign_tot if sign_tot else float("nan"))


def prefill_buffer(net, cfg, buffer, val_buffer, pool, rng, opponent_pool, target):
    """Fill the replay buffer with fresh self-play from the CURRENT champion before
    the first training step.

    A --resume starts with an empty buffer: `fill_frac` then throttles the step
    count for the ~4 rounds it takes to refill, so every restart wastes rounds,
    and the rounds that do train see a thin, low-diversity buffer. Prefilling pays
    that self-play cost once, up front, in one generation of weights — which is
    exactly the data distribution the buffer is supposed to hold.

    Runs in chunks of cfg.games_per_iter so progress is visible and the held-out
    split (cfg.val_frac) is applied the same way the training loop applies it.
    """
    if target <= 0 or len(buffer) >= target:
        return
    print(f"[prefill] filling replay buffer to {target} examples "
          f"(have {len(buffer)}) with self-play from the current champion")
    t0 = time.time()
    games = 0
    chunk = 0
    while len(buffer) < target:
        # a distinct seed space from the training loop's (cfg.seed + it * 100000),
        # so prefill games are not replays of iteration 1..N games.
        examples, _ = selfplay_parallel(
            net, cfg, cfg.games_per_iter, base_seed=cfg.seed + 900_000_000 + chunk * 100000,
            pool=pool, n_workers=cfg.num_workers, opponent_pool=opponent_pool)
        n_val = int(len(examples) * cfg.val_frac)
        if n_val:
            perm = rng.permutation(len(examples))
            val_buffer.add_many([examples[i] for i in perm[:n_val]])
            examples = [examples[i] for i in perm[n_val:]]
        buffer.add_many(expand_symmetries(examples, cfg))
        games += cfg.games_per_iter
        chunk += 1
        print(f"[prefill] {len(buffer)}/{target} examples after {games} games "
              f"({time.time() - t0:.0f}s)")
    print(f"[prefill] done: {len(buffer)} examples, {len(val_buffer)} held out, "
          f"{games} games, {time.time() - t0:.0f}s")


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


def run(cfg, resume=False, references_dir=None, init_from=None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cpu = torch.device("cpu")
    train_device = resolve_device(cfg.device)
    print(f"[cfg] board={cfg.board_size}x{cfg.board_size} net={cfg.res_blocks}x{cfg.channels} "
          f"sims={cfg.num_simulations} games/iter={cfg.games_per_iter} "
          f"train_device={train_device} selfplay=cpu")

    metrics.ensure_dir(cfg)

    net = StackNet(cfg.board_size, cfg.channels, cfg.res_blocks,
                   cfg.policy_head)
    # --init-from: start self-play from a net trained elsewhere (e.g. supervised
    # imitation of AlphaBeta). Weights only -- no version registry, no match
    # history, no iteration counter -- so this is a fresh run with a warm prior,
    # which is what --resume is NOT. The architecture comes from the checkpoint
    # rather than from cfg, because a mismatched head silently fails to load.
    if init_from:
        pre = torch.load(init_from, map_location="cpu", weights_only=False)
        net = build_net(pre["arch"], pre["state"])
        net.load_state_dict(pre["state"])
        cfg.channels, cfg.res_blocks = net.arch()["channels"], net.arch()["res_blocks"]
        cfg.policy_head = net.policy_head
        print(f"[init] {init_from} -> {cfg.res_blocks}x{cfg.channels} "
              f"{cfg.policy_head}-head, extra={pre.get('extra', {})}")
    # AdamW, not Adam: Adam folds weight_decay into the gradient, and its per-parameter
    # scaling then decays low-gradient weights far harder than intended. AdamW decouples it.
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer = ReplayBuffer(cfg.replay_capacity)
    val_buffer = ReplayBuffer(cfg.val_capacity)   # held out, never trained on
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
    ab_history = []                         # winrate of each AlphaBeta match played
    ab_beaten = False                       # True once AB stops being informative

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
                    ab_history = [r["winrate_vs_ab_light"] for r in rows
                                  if r.get("winrate_vs_ab_light") is not None]
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

    elo = (compute_elo(book, anchor=cfg.elo_anchor, anchor_elo=cfg.elo_anchor_value,
                       prior_draws=cfg.elo_prior_draws) if book.d else {})
    # the reference tracked in the win-rate chart (strongest one; one is enough)
    primary_ref = max(ref_ids, key=lambda i: elo.get(i, -1e9)) if ref_ids else None
    # gate/gauntlet/benchmark matches run at reduced sims — self-play keeps full sims
    ecfg = replace(cfg, num_simulations=cfg.eval_simulations)

    try:
      if cfg.prefill_frac > 0:
        net.to(cpu)
        metrics.write_status(cfg, {"iter": start_iter, "phase": "prefill",
                                   "games": 0, "buffer": len(buffer)})
        past = champion_ids[:-1]
        pre_pool = [_load_ckpt(registry[i]["path"])
                    for i in (past[-cfg.pool_size:] if cfg.pool_size > 0 else [])
                    if i in registry]
        prefill_buffer(net, cfg, buffer, val_buffer, pool, rng, pre_pool,
                       int(cfg.prefill_frac * cfg.replay_capacity))

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
        # Hold out a slice of the FRESH positions before augmentation, so no
        # symmetry of a validation position can leak into the training buffer.
        n_val = int(len(examples) * cfg.val_frac)
        if n_val:
            perm = rng.permutation(len(examples))
            val_buffer.add_many([examples[i] for i in perm[:n_val]])
            examples = [examples[i] for i in perm[n_val:]]
        aug = expand_symmetries(examples, cfg)
        buffer.add_many(aug)
        added = len(aug)
        if last_record is not None:
            metrics.write_last_game(cfg, {**last_record, "iter": it})
        sp_time = time.time() - t0

        # ------------------------------------------------------ train (GPU)
        metrics.write_status(cfg, {"iter": it, "phase": "train", "buffer": len(buffer)})
        t_train = time.time()
        net.to(train_device)
        # Snapshot the PREVIOUS best (weights + optimizer momentum) BEFORE training,
        # so a rejected candidate can be fully reverted — net AND optimizer state.
        # (Snapshotting after training would capture the candidate, not the best.)
        prev_best_state = copy.deepcopy(net.state_dict()) if cfg.use_gate else None
        prev_best_opt = copy.deepcopy(opt.state_dict()) if cfg.use_gate else None
        # cosine LR decay from cfg.lr down to cfg.lr_min across the whole run
        frac = min(it, cfg.iterations) / max(cfg.iterations, 1)
        lr_now = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * frac))
        for gparam in opt.param_groups:
            gparam["lr"] = lr_now
        # Scale steps to how FULL the buffer is, not just "bigger than one batch".
        # A cold/post-resume buffer is thin (few distinct games), so the full step
        # count reuses each position many times in one round and overfits to it —
        # measured after the 2026-08-18 resume: value train MSE fell 0.80->0.56
        # while held-out ROSE 0.91->1.00 during the 4 rounds it took the buffer to
        # refill. Ramping steps with fill_frac makes early post-resume rounds train
        # lightly (like a fresh run naturally does) instead of overtraining on thin data.
        fill_frac = min(1.0, len(buffer) / cfg.replay_capacity)
        steps = int(cfg.train_steps_per_iter * fill_frac) if len(buffer) >= cfg.batch_size else 0
        if steps:
            pol_loss, val_loss, own_loss = train_net(net, opt, buffer, cfg, train_device, steps)
        else:
            pol_loss = val_loss = own_loss = float("nan")
        v_pol, v_val, v_top1, v_sign = eval_net(net, val_buffer, cfg, train_device)
        tr_time = time.time() - t_train

        # ---------------- gate (normal AlphaZero) + benchmarks incl. reference
        net.to(cpu)
        base = cfg.seed + it * 100000
        cand_wr = None
        accepted = False
        wr_random = wr_ref = 0.0
        wr_ab_light = None
        cand_elo = None                    # None on non-eval iterations -> chart skips it
        best_elo = elo.get(f"v{best_iter}", 0.0)
        opponent_iter = best_iter          # previous best this candidate is gated against

        # Evaluation is expensive (at eval_every=1 the match games cost about as much
        # as self-play itself). Run the whole block only every Nth iteration so the
        # clock goes into learning; Elo/gate points simply appear less often.
        do_eval = steps and (it % cfg.eval_every == 0 or it == cfg.iterations)
        # The Elo matches (vs previous best + the gauntlet) run on only every Nth
        # eval block, but with many more games each. A 4-game sample cannot tell
        # two nets 3 rounds apart apart at all: the chart drifted DOWN over 50
        # rounds while a 40-game v156-vs-v105 match went 28-12 for the newer net.
        # Same total game budget, ~5x the resolution, far fewer misleading points.
        # `best_iter == 0` forces the FIRST eval block to play the previous best,
        # which is still v0. Without it the very first candidate has no game
        # against v0, compute_elo cannot find its anchor, and every rating on the
        # chart floats on an arbitrary scale until the first scheduled gauntlet.
        do_gauntlet = do_eval and ((it // cfg.eval_every) % cfg.gauntlet_every == 0
                                   or best_iter == 0
                                   or it == cfg.iterations)
        if do_eval:
            metrics.save_version(cfg, net, it)
            registry[cand_id] = {"path": os.path.join(cfg.ckpt_dir, "versions", f"v{it}.pt"),
                                 "arch": net.arch(), "iter": it, "ref": False}
            version_ids.append(cand_id)
            a_arch, a_state = net_arch_state(net)

            # Candidate vs the PREVIOUS BEST. With cfg.use_gate it decides
            # acceptance (classic AlphaZero); without it, it is only a progress
            # line + the Elo chain link to the immediate predecessor, and every
            # candidate is accepted (modern AlphaZero / KataGo). Gating threw
            # away ~2/3 of all training on the previous run: once the true gain
            # per iteration drops below the 20-game gate noise, accept/reject is
            # a coin flip and each reject un-does a full iteration of SGD.
            # eval matches run at reduced sims (ecfg) — same for both sides, so fair.
            metrics.write_status(cfg, {"iter": it, "phase": "gate", "buffer": len(buffer)})
            n_vs_best = cfg.eval_games if cfg.use_gate else cfg.gauntlet_games
            if cfg.use_gate or do_gauntlet:
                wa, wb, dr = match_parallel(a_arch, a_state, _az_opp(registry, f"v{best_iter}"),
                                            ecfg, n_vs_best, base + 50000, pool, cfg.num_workers)
                cand_wr = win_rate(wa, wb, dr)
                book.add_match(cand_id, f"v{best_iter}", wa, wb, dr)
            accepted = (cand_wr >= cfg.eval_win_threshold) if cfg.use_gate else True

            metrics.write_status(cfg, {"iter": it, "phase": "benchmark", "buffer": len(buffer)})
            if cfg.benchmark_games > 0:          # off by default — the net sweeps random
                wa, wb, dr = match_parallel(a_arch, a_state, ("random", None), ecfg,
                                            cfg.benchmark_games, base + 70000, pool, cfg.num_workers)
                wr_random = win_rate(wa, wb, dr)
                book.add_match(cand_id, "random", wa, wb, dr)

            # Fixed-budget AlphaBeta: the one opponent from outside our own lineage,
            # rated in the SAME Elo pool so it draws as a bar on the chart. Once the
            # net beats it reliably it stops carrying information (same trap `random`
            # fell into), so we stop playing it every iteration and only re-check
            # every `ab_recheck_every` — enough to keep its bar linked to the current
            # versions without paying for it forever.
            play_ab = cfg.ab_elo_games > 0 and (not ab_beaten
                                                or it % cfg.ab_recheck_every == 0)
            if play_ab:
                wa, wb, dr = match_parallel(a_arch, a_state, ("alphabeta", cfg.alphabeta_budget),
                                            ecfg, cfg.ab_elo_games, base + 73000, pool, cfg.num_workers)
                wr_ab_light = win_rate(wa, wb, dr)
                ab_history.append(wr_ab_light)
                book.add_match(cand_id, "alphabeta", wa, wb, dr)
                win = cfg.ab_stop_window
                recent = (sum(ab_history[-win:]) / win) if len(ab_history) >= win else 0.0
                if recent >= cfg.ab_stop_winrate and not ab_beaten:
                    ab_beaten = True
                    print(f"[ab] AlphaBeta({cfg.alphabeta_budget}s) beaten "
                          f"({recent:.0%} over last {win}) — from now on only re-checked "
                          f"every {cfg.ab_recheck_every} iterations")
                elif recent < cfg.ab_stop_winrate and ab_beaten:
                    ab_beaten = False
                    print(f"[ab] AlphaBeta({cfg.alphabeta_budget}s) back in play "
                          f"({recent:.0%} over last {win})")

            if primary_ref is not None:
                wa, wb, dr = match_parallel(a_arch, a_state, _az_opp(registry, primary_ref), ecfg,
                                            cfg.benchmark_games, base + 77000, pool, cfg.num_workers)
                wr_ref = win_rate(wa, wb, dr)
                book.add_match(cand_id, primary_ref, wa, wb, dr)

            # gauntlet: candidate vs a spread of past versions (Elo connectivity)
            if do_gauntlet:
                others = [i for i in version_ids if i not in (cand_id, f"v{best_iter}")]
                for k, oid in enumerate(sample_ids(others, cfg.gauntlet_versions, rng)):
                    wa, wb, dr = match_parallel(a_arch, a_state, _az_opp(registry, oid), ecfg,
                                                cfg.gauntlet_games, base + 80000 + k * 1000,
                                                pool, cfg.num_workers)
                    book.add_match(cand_id, oid, wa, wb, dr)

            elo = compute_elo(book, anchor=cfg.elo_anchor, anchor_elo=cfg.elo_anchor_value,
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
            "anchor": cfg.elo_anchor, "anchor_value": cfg.elo_anchor_value,
            "ab_budget": cfg.alphabeta_budget,
            "best_iter": best_iter, "primary_ref": primary_ref,
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
            "elo": None if cand_elo is None else round(cand_elo, 1),
            "best_elo": round(best_elo, 1),
            "elo_alphabeta": 0.0,               # random is the anchor now
            "lr": round(lr_now, 6),
            "policy_loss": round(pol_loss, 4),
            "value_loss": round(val_loss, 4),
            "own_loss": round(own_loss, 4),
            # held-out (never trained on): the train/val gap is the overfitting signal
            "val_policy_loss": None if v_pol != v_pol else round(v_pol, 4),
            "val_value_loss": None if v_val != v_val else round(v_val, 4),
            "policy_top1": None if v_top1 != v_top1 else round(v_top1, 4),
            "value_sign_acc": None if v_sign != v_sign else round(v_sign, 4),
            "val_positions": len(val_buffer),
            "cand_winrate_vs_best": None if cand_wr is None else round(cand_wr, 3),
            "accepted": accepted,
            "winrate_vs_random": round(wr_random, 3) if cfg.benchmark_games else None,
            "winrate_vs_ab_light": wr_ab_light if wr_ab_light is None else round(wr_ab_light, 3),
            "winrate_vs_alphabeta": None if ab_wr is None else round(ab_wr, 3),
            "winrate_vs_reference": round(wr_ref, 3),
            "reference_id": None if primary_ref is None else primary_ref.replace("ref:", ""),
            "buffer": len(buffer),
            "new_examples": added,
            "selfplay_sec": round(sp_time, 1),
            "train_sec": round(tr_time, 1),
            # Everything left over is the match games (vs previous best, gauntlet,
            # AlphaBeta). This is the number the 80/20 learning-vs-measuring target
            # is judged on — before, it was only ever estimated, never recorded.
            "eval_sec": round(max(0.0, time.time() - t_iter - sp_time - tr_time), 1),
            "iter_sec": round(time.time() - t_iter, 1),
        }
        metrics.append_metric(cfg, row)
        print(f"[iter {it:3d}] v{it} elo={'-' if cand_elo is None else f'{cand_elo:.0f}'} "
              f"loss p={pol_loss:.3f}/val {v_pol:.3f} v={val_loss:.3f}/val {v_val:.3f} "
              f"top1={v_top1:.2f} vsign={v_sign:.2f} "
              f"vs_prev={'-' if cand_wr is None else f'{cand_wr:.2f}'} "
              f"vs_AB={'-' if wr_ab_light is None else f'{wr_ab_light:.2f}'} "
              f"vs_AB5s={'-' if ab_wr is None else f'{ab_wr:.2f}'}"
              f"{'' if accepted or not (do_eval and cfg.use_gate) else ' (rej)'} "
              f"{row['iter_sec']}s "
              f"(sp {row['selfplay_sec']} / tr {row['train_sec']} / ev {row['eval_sec']})")

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
    ap.add_argument("--channels", type=int, default=d.channels)
    ap.add_argument("--blocks", type=int, default=d.res_blocks, dest="res_blocks")
    ap.add_argument("--eval-every", type=int, default=d.eval_every, dest="eval_every")
    ap.add_argument("--train-steps", type=int, default=d.train_steps_per_iter,
                    dest="train_steps_per_iter")
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None, dest="init_from",
                    help="path to a .pt whose weights seed the net. Fresh run "
                         "(iter 0, empty history), warm prior. Arch is read from "
                         "the checkpoint. Not compatible with --resume.")
    ap.add_argument("--q-ratio", type=float, default=None, dest="value_q_ratio",
                    help="weight of the search's own value in the value target "
                         "(0 = pure game result, lc0-style blend at ~0.5)")
    ap.add_argument("--prefill", type=float, default=None, dest="prefill_frac",
                    help="before the first training step, self-play until the replay "
                         "buffer is this full (0-1). 1.0 = completely full. Use on "
                         "--resume: the buffer is not persisted across restarts.")
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
    ap.add_argument("--gate", action="store_true",
                    help="re-enable the classic accept/revert gate (off by default)")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = Config(iterations=a.iterations, games_per_iter=a.games_per_iter,
                 num_simulations=a.num_simulations, board_size=a.board_size,
                 channels=a.channels, res_blocks=a.res_blocks,
                 eval_every=a.eval_every,
                 train_steps_per_iter=a.train_steps_per_iter,
                 device=a.device, seed=a.seed)
    # ablation / experiment overrides
    if a.ckpt_dir is not None:
        cfg.ckpt_dir = a.ckpt_dir
        cfg.metrics_file = os.path.join(a.ckpt_dir, "metrics.jsonl")
        cfg.ab_bench_file = os.path.join(a.ckpt_dir, "ab_bench.jsonl")
    if a.num_workers is not None:
        cfg.num_workers = a.num_workers
    if a.prefill_frac is not None:
        cfg.prefill_frac = max(0.0, min(1.0, a.prefill_frac))
    if a.value_q_ratio is not None:
        cfg.value_q_ratio = max(0.0, min(1.0, a.value_q_ratio))
    if a.no_gumbel:
        cfg.self_play_gumbel = False
    if a.pcr_prob is not None:
        cfg.pcr_prob = a.pcr_prob
    if a.own_loss_weight is not None:
        cfg.own_loss_weight = a.own_loss_weight
    if a.no_tree_reuse:
        cfg.tree_reuse = False
    if a.gate:
        cfg.use_gate = True
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
    if a.init_from and a.resume:
        raise SystemExit("--init-from and --resume are mutually exclusive: one "
                         "starts a fresh run from given weights, the other "
                         "continues an old run.")
    run(cfg, resume=a.resume, references_dir=a.references_dir,
        init_from=a.init_from)


if __name__ == "__main__":
    main()
