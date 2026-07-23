"""Hyperparameters for StackIt AlphaZero, in one place.

Defaults are tuned for a *fast local run on a 4x4 board* — enough to watch the
net climb from random to beating AlphaBeta within an evening. Scale `board_size`
up (and sims/games/net with it) once the 4x4 pipeline is proven.
"""
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # --- game ---
    board_size: int = 4          # NxN. Start small; 16 actions on 4x4.
    max_game_plies: int = 60     # self-play move cap; decided by box count if hit.

    # --- network ---
    channels: int = 96           # filters in the residual tower
    res_blocks: int = 5          # number of residual blocks

    # --- MCTS ---
    num_simulations: int = 96    # rollouts (net evals) per move — biggest quality knob
    c_puct: float = 1.5          # exploration constant in PUCT
    dirichlet_alpha: float = 0.6 # root noise concentration (~10/avg_moves)
    dirichlet_eps: float = 0.30  # root noise weight (self-play only)
    temp_moves: int = 16         # plies of tau=1 sampling before switching to argmax
    opening_random_plies: int = 3  # uniformly-random opening moves per self-play game

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
    eval_every: int = 1          # run arena/benchmarks every N iterations
    eval_games: int = 20         # candidate-vs-best arena games
    eval_win_threshold: float = 0.55  # promote candidate to "best" at >= this
    benchmark_games: int = 16    # games vs random / alphabeta for absolute signal
    alphabeta_budget: float = 0.05    # seconds/move for the AlphaBeta benchmark opponent
    gauntlet_versions: int = 4   # how many past versions the best plays each iter (Elo)
    gauntlet_games: int = 4      # games vs each sampled past version
    elo_prior_draws: float = 2.0  # virtual draws per pair — tames 100%-sweep Elo blow-ups

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
