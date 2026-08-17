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
    max_game_plies: int = 90     # self-play move cap; decided by box count if hit.

    # --- network ---
    channels: int = 96           # filters in the residual tower
    res_blocks: int = 6          # number of residual blocks

    # --- MCTS ---
    num_simulations: int = 256   # rollouts (net evals) per move — biggest quality knob
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
    gumbel_c_scale: float = 1.0  # sigma(q) magnitude scale
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
    games_per_iter: int = 40     # self-play games generated each iteration
    replay_capacity: int = 100_000   # samples kept in the replay buffer (post-augment)
    train_steps_per_iter: int = 400  # SGD minibatches per iteration
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4   # L2 regularization
    augment_symmetries: bool = True  # 8-fold dihedral augmentation (square boards)

    # --- evaluation / gating ---
    eval_simulations: int = 128  # sims for gate/gauntlet/benchmark matches (self-play keeps num_simulations)
    az_time_budget: float = 0.0  # if >0, AZPlayer searches by wall-clock (s/move) instead of fixed sims
    eval_games: int = 20         # candidate-vs-best arena games
    eval_win_threshold: float = 0.55  # promote candidate to "best" at >= this
    benchmark_games: int = 16    # games vs random for absolute signal
    alphabeta_budget: float = 0.3     # seconds/move for the fixed-budget AlphaBeta rating opponent
    ab_elo_games: int = 8        # games vs fixed-budget AlphaBeta each iter (feeds its Elo)
    gauntlet_versions: int = 4   # how many past versions the best plays each iter (Elo)
    gauntlet_games: int = 4      # games vs each sampled past version
    elo_prior_draws: float = 2.0  # virtual draws per pair — tames 100%-sweep Elo blow-ups

    # --- background "strong AlphaBeta" benchmark (decoupled from the training loop) ---
    ab_bench_every: int = 5      # launch the strong-AB benchmark every Nth accepted champion
    ab_bench_think: float = 5.0  # seconds/move for BOTH AlphaBeta and the net in this benchmark
    ab_bench_games: int = 8      # games per background benchmark
    ab_bench_workers: int = 4    # CPU processes for it (leave cores for training)
    ab_bench_file: str = "checkpoints/ab_bench.jsonl"

    # --- io ---
    ckpt_dir: str = "checkpoints"
    metrics_file: str = "checkpoints/metrics.jsonl"
    device: str = "auto"         # TRAIN device. auto -> mps/cuda if available else cpu
    selfplay_device: str = "cpu"  # MCTS is batch-1 -> CPU beats MPS on tiny nets
    num_workers: int = 8         # CPU processes for parallel self-play/eval games
    seed: int = 0

    def to_dict(self):
        return asdict(self)


DEFAULT = Config()
