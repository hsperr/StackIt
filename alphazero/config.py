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
    channels: int = 64           # filters in the residual tower. Went 64->96 on 2026-08-19
                                 # (the net disagreed with ITSELF on 48% of rotated positions
                                 # despite 8-fold augmentation) and back to 64 later the same
                                 # day: that arm was stopped at v30, and the diagnosis moved
                                 # from capacity to self-play COVERAGE — see gumbel_noise_plies.
                                 # 64x4 also iterates ~2.3x faster, which matters more while
                                 # the algorithm is still being debugged.
    res_blocks: int = 4          # number of residual blocks

    # --- MCTS ---
    num_simulations: int = 256   # rollouts (net evals) per move — biggest quality knob. Halved to
                                 # 128 on 2026-08-18 to buy games; doubled back to 256 on
                                 # 2026-08-19. At gumbel_m=16 the old setting gave only 8 sims per
                                 # root action, so the search barely improved on the raw policy —
                                 # and the search IS the training target. Playout Cap
                                 # Randomization softens the cost: only pcr_prob of moves get the
                                 # full budget, so the real per-game cost rises ~1.6x, not 2x.
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
                                   # tau=1 sampling (the pre-Gumbel algorithm that converged).
                                   # Verified 2026-08-19 against DeepMind's mctx: the completed
                                   # policy IS the documented training target ("action_weights
                                   # contain targets usable to train the policy probabilities"),
                                   # and it is MEANT to be sharper than visit counts. Do not
                                   # "fix" that by reverting to visit counts.
    gumbel_noise_plies: int = 0  # plies of root Gumbel noise in self-play; 0 = every ply, which
                                 # is what mctx does (gumbel_scale=1.0 throughout training, 0
                                 # only for evaluation). Was capped at temp_moves=16, so from
                                 # ply 16 on self-play was deterministic and every recorded
                                 # position sat on the net's own greedy line.

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
    train_steps_per_iter: int = 400   # SGD minibatches per iteration. Tried raising 400->1200 at
                                 # round 34 of the 2026-08-18 run (hypothesis: undertrained, not
                                 # overfitted — see git history for the reasoning). Result once the
                                 # buffer refilled cleanly (round 46 on): value gap grew to +0.36
                                 # (was +0.11), held-out policy loss WORSE (2.21 vs 2.09), top1
                                 # WORSE (0.29 vs 0.35) — train falls, held-out doesn't: stale/
                                 # disagreeing targets across buffer generations, not undertraining.
                                 # Reverted to 400. If retried, fix the buffer instead (KataGo's
                                 # growing window, or MuZero Reanalyse) rather than more steps.
    batch_size: int = 128
    value_q_ratio: float = 0.5   # weight of the SEARCH's own value in the value target;
                                 # 0 = pure game result z (AlphaZero/Leela Zero), 1 = pure
                                 # search value Q. lc0 calls this q_ratio and trains on a
                                 # blend. Why: z gives ONE label per game, copied onto every
                                 # recorded position of that game, so 100k rows carry only
                                 # ~800 distinct answers and an even position gets stamped
                                 # with however the game happened to end. Q is the root's
                                 # own averaged search value, so every position carries its
                                 # own answer. Keep some z in the mix — it is the only label
                                 # that is not the net grading its own homework.
    prefill_frac: float = 0.0    # before the FIRST training step, generate self-play with the
                                 # loaded champion until the replay buffer is this full
                                 # (0 = off, 1.0 = full). A resume starts with an EMPTY buffer:
                                 # the fill_frac ramp below then trains lightly for ~4 rounds on
                                 # thin data, and every restart costs that. Prefilling pays the
                                 # self-play cost once, up front, and the first trained round
                                 # already sees a full, single-generation buffer.
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
    alphabeta_engine: str = "c"       # "c" = the C engine in c_engine/ (~50x the nodes/sec of the
                                      # Python one, and it has the stand-pat quiescence fix);
                                      # "python" = the in-process AlphaBeta.
    alphabeta_budget: float = 0.05    # seconds/move for the fixed-budget AlphaBeta rating opponent.
                                      # Dropped 0.3->0.05 because the C engine at 0.3s reaches ~9.5
                                      # ply and would simply shut out a fresh net, which measures
                                      # nothing. 0.05s is still ~6-7 ply — a real ladder rung.
    ab_elo_games: int = 8        # games vs fixed-budget AlphaBeta each iter (feeds its Elo)
    ab_stop_winrate: float = 0.75  # once the mean winrate over the last `ab_stop_window`
    ab_stop_window: int = 3        # AlphaBeta matches reaches this, stop playing it every
    ab_recheck_every: int = 25     # iteration — it is beaten. Re-play it every Nth iteration
                                   # so its Elo bar stays linked to the current versions.
    gauntlet_versions: int = 2   # how many past versions the best plays (Elo connectivity)
    gauntlet_games: int = 20     # games vs each sampled past version. Was 4, which cannot
                                 # resolve two nets 3 rounds apart: over 50 rounds the Elo chart
                                 # drifted DOWN while a 40-game match between v156 and v105 went
                                 # 28-12 for the newer net. Same total cost, ~5x the resolution.
    gauntlet_every: int = 5      # run the gauntlet only every Nth EVAL block (so with
                                 # eval_every=3, every 15 iterations). Rare and big beats
                                 # frequent and noisy — 4-game samples are pure coin-flip.
    match_random_plies: int = 3  # uniformly random opening moves in EVERY arena game. Self-play
                                 # already did this; arena games did not, so the same two nets
                                 # replayed nearly the same game N times and an "N-game match"
                                 # carried far less than N games of information.
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
    ab_bench_think: float = 1.0  # seconds/move for BOTH AlphaBeta and the net in this benchmark.
                                 # 5.0 was chosen to make the SLOW Python engine strong; the C
                                 # engine at 1.0s already searches deeper than Python did at 5.0s,
                                 # so this is a harder opponent that costs a fifth of the clock.
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
