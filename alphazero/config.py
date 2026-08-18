"""Hyperparameters for StackIt AlphaZero, in one place.

Defaults are tuned for a local run on a 5x5 board — enough to watch the net climb
from random to beating AlphaBeta. Scale `board_size` up (and sims/games/net with
it) once the 5x5 pipeline is proven.
"""
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # --- game ---
    board_size: int = 5          # NxN. 25 actions on 5x5.
    max_game_plies: int = 400    # self-play move cap; decided by box count if hit. Measured
                                 # 2026-08-18 on 5x5: at 90 a THIRD of games were cut off and got
                                 # a box-count winner (a different rule) — poisoning value labels.
                                 # Given room every game ends by domination: median 74, max 364.

    # --- network ---
    channels: int = 64           # filters in the residual tower
    res_blocks: int = 4          # number of residual blocks

    # --- MCTS ---
    num_simulations: int = 128   # rollouts (net evals) per move — biggest quality knob. Halved
                                 # 2026-08-18 to buy games: each game is ~1.6x cheaper, and games
                                 # (not positions) are what feeds the value head. See games_per_iter.
    c_puct: float = 1.5          # exploration constant in PUCT (interior nodes)
    dirichlet_alpha: float = 0.6 # root noise concentration (~10/avg_moves)
    dirichlet_eps: float = 0.30  # root noise weight (classic/time-budget path only)
    temp_moves: int = 16         # plies of tau=1 sampling before switching to argmax
    opening_random_plies: int = 3  # uniformly-random opening moves per self-play game

    # --- Gumbel root (fixed-sim searches: self-play + gate/eval) ---
    # Gumbel-top-k + Sequential Halving at the root. Guarantees a policy
    # improvement even at low sim counts (the fix for the gate plateau), and
    # replaces Dirichlet noise as the self-play exploration device. The
    # wall-clock (time_budget) path keeps classic PUCT.
    gumbel_m: int = 16           # actions considered at the root (top-m by logit+Gumbel)
    gumbel_c_visit: float = 50.0 # sigma(q) visit scale — how much Q shifts the policy
    gumbel_c_scale: float = 0.1  # sigma(q) magnitude scale (mctx `value_scale`;
                                 # applied to Q rescaled to [0,1] — see mcts_az._sigma.
                                 # 1.0 on raw Q made targets near-one-hot noise.)
    self_play_gumbel: bool = True  # ABLATION: False -> classic PUCT visit-count target +
                                   # tau=1 sampling (the pre-Gumbel algorithm that converged)

    # --- Playout Cap Randomization (KataGo) ---
    # Most self-play moves get a cheap search and are NOT recorded; a fraction
    # get a full search and ARE recorded. More games/hour, cleaner targets.
    pcr_prob: float = 0.25       # fraction of main-net moves that are full-search + recorded
    pcr_fast_sims: int = 32      # sims for the cheap (unrecorded) moves
    tree_reuse: bool = True      # reuse the chosen child's subtree across self-play plies

    # --- auxiliary ownership head (KataGo's biggest single trick) ---
    # Predict each cell's final owner (empty/mine/theirs) — a dense per-cell
    # signal on top of the single scalar value. Big learning-efficiency win.
    own_loss_weight: float = 0.15  # weight of the ownership cross-entropy in the loss

    # --- opponent pool (data diversity + robustness) ---
    pool_size: int = 2           # how many recent champions to mix into self-play
    pool_play_frac: float = 0.4  # fraction of self-play games vs a pool/reference opponent

    # --- self-play / training loop ---
    iterations: int = 200        # outer policy-iteration rounds
    games_per_iter: int = 200    # self-play games generated each iteration. Each game yields
                                 # exactly ONE win/lose label, so this — not position count —
                                 # is what feeds the value head. 40 was starving it; at 70 the
                                 # value head still memorised (train MSE 0.53 vs held-out 1.02
                                 # over rounds 2-9 of the 2026-08-18 run) because ~9000 training
                                 # rows carried only 70 distinct answers.
    replay_capacity: int = 100_000   # samples kept in the replay buffer (post-augment)
    train_steps_per_iter: int = 1200  # SGD minibatches per iteration. Raised from 400 at round 34
                                 # of the 2026-08-18 run: policy CE sat at 2.06 train / 2.09 held-out
                                 # against a MEASURED target-entropy floor of 0.91 — 1.15 above the
                                 # floor with a 0.03 generalisation gap, i.e. undertrained, not
                                 # overfitted. Training was only 3% of the wall clock.
    batch_size: int = 128
    lr: float = 1e-3             # peak LR; cosine-decayed to lr_min over `iterations`
    lr_min: float = 1e-4
    grad_clip: float = 10.0      # max global grad norm (0 disables). Measured 2026-08-18 on the
                                 # iter-74 5x5 champion: real grad norm median 5.53 / p90 5.83, so
                                 # the old 1.0 clipped EVERY step and silently scaled the effective
                                 # LR down ~5.5x. 10.0 only catches genuine spikes.
    weight_decay: float = 1e-4   # L2 regularization
    augment_symmetries: bool = True  # 8-fold dihedral augmentation (square boards)
    val_frac: float = 0.1        # fraction of each iteration's fresh self-play positions held
    val_capacity: int = 4000     # OUT of the replay buffer, never trained on. Gives a real
                                 # held-out loss (train-vs-val gap = overfitting to the buffer)
                                 # plus top-1 policy agreement and value sign accuracy.

    # --- evaluation / gating ---
    eval_every: int = 3          # run the whole evaluation block (vs previous, gauntlet,
                                 # AlphaBeta) only every Nth iteration. Measuring cost about
                                 # matched self-play cost at eval_every=1 (53% of the clock);
                                 # 3 puts it near 20%, so the rest goes into learning.
    eval_simulations: int = 128  # sims for gate/gauntlet/benchmark matches (self-play keeps num_simulations)
    az_time_budget: float = 0.0  # if >0, AZPlayer searches by wall-clock (s/move) instead of fixed sims
    use_gate: bool = False       # False = modern AlphaZero/KataGo: every candidate
                                 # becomes the new best, nothing is ever reverted.
                                 # True = classic gate (throws away ~2/3 of training
                                 # once per-iteration gains fall under gate noise).
    eval_games: int = 20         # candidate-vs-best arena games (gate only)
    eval_win_threshold: float = 0.55  # promote candidate to "best" at >= this
    benchmark_games: int = 0     # games vs random (0 = off; the net sweeps random from
                                 # iter ~7, so it measures nothing and just costs time)
    alphabeta_budget: float = 0.3     # seconds/move for the fixed-budget AlphaBeta rating opponent
    ab_elo_games: int = 8        # games vs fixed-budget AlphaBeta each iter (feeds its Elo)
    ab_stop_winrate: float = 0.75  # once the mean winrate over the last `ab_stop_window`
    ab_stop_window: int = 3        # AlphaBeta matches reaches this, stop playing it every
    ab_recheck_every: int = 25     # iteration — it is beaten. Re-play it every Nth iteration
                                   # so its Elo bar stays linked to the current versions.
    gauntlet_versions: int = 3   # how many past versions the best plays each iter (Elo)
    gauntlet_games: int = 4      # games vs each sampled past version
    elo_prior_draws: float = 2.0  # virtual draws per pair — tames 100%-sweep Elo blow-ups
    elo_anchor: str = "v0"       # Which participant defines the scale. v0 = the initial
    elo_anchor_value: float = 1000.0  # random-weights net, pinned at 1000 — the same
                                 # convention Lc0/KataGo use (first net = 0), shifted so it
                                 # reads like chess Elo. The anchor must NEVER move: pin it
                                 # to a changing player (e.g. "previous best") and every
                                 # point lands near the anchor and the chart goes flat.
                                 # AlphaBeta is drawn as a horizontal bar instead.

    # --- background "strong AlphaBeta" benchmark (decoupled from the training loop) ---
    ab_bench_every: int = 10     # launch the strong-AB benchmark every Nth accepted champion
                                 # (at 5s/move it runs for ~10min; firing it every 5
                                 # iterations kept it permanently resident and stole cores)
    ab_bench_think: float = 5.0  # seconds/move for BOTH AlphaBeta and the net in this benchmark
    ab_bench_games: int = 8      # games per background benchmark
    ab_bench_workers: int = 2    # CPU processes for it (leave cores for training)
    ab_bench_file: str = "checkpoints/ab_bench.jsonl"

    # --- io ---
    ckpt_dir: str = "checkpoints"
    metrics_file: str = "checkpoints/metrics.jsonl"
    device: str = "auto"         # TRAIN device. auto -> mps/cuda if available else cpu
    selfplay_device: str = "cpu"  # MCTS is batch-1 -> CPU beats MPS on tiny nets
    num_workers: int = 10        # CPU processes for parallel self-play/eval games
    seed: int = 0

    def to_dict(self):
        return asdict(self)


DEFAULT = Config()
