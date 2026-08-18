"""Generate self-play games with MCTS and turn them into training examples.

Each recorded position stores (encoded_state, pi, player), where pi is the
Gumbel completed-policy improvement target. When the game ends, z (the outcome
from that position's player's view) and a per-cell ownership target (which player
owns each cell at game end, from that position's player's view) are filled in.

Two efficiency devices from KataGo / Gumbel AlphaZero are used here:

* Playout Cap Randomization: only a fraction (`pcr_prob`) of the main net's
  moves get a full search and are recorded; the rest get a cheap `pcr_fast_sims`
  search and are played but NOT recorded. More games/hour, cleaner targets.
* Tree reuse: in pure self-play the chosen child's subtree is carried into the
  next ply, so its search starts from work already done.

Each ply plays the Gumbel search's *selected action* (the Sequential-Halving
survivor), while recording the completed policy as the training target. Root
Gumbel noise is the exploration device: on for the first `temp_moves` plies (so
the selected action varies), off afterwards (greedy).
"""
from dataclasses import replace

import numpy as np

from board import Board
from .encoding import encode, index_to_move, augment
from .mcts_az import MCTS, terminal_value


def game_winner(board):
    """0 draw / 1 / 2 — domination first, else box count (arena rules)."""
    w = board.winning_player()
    if w:
        return w
    b1, b2 = board.boxes_for(1), board.boxes_for(2)
    return 1 if b1 > b2 else 2 if b2 > b1 else 0


def _ownership(final_board, mover, n):
    """Flat len-N*N int array of each cell's final owner from `mover`'s view:
    0 empty, 1 mine, 2 theirs. Indexed y*N+x to match action indices."""
    own = np.zeros(n * n, dtype=np.int64)
    grid = final_board.player
    for y in range(n):
        row = grid[y]
        for x in range(n):
            o = row[x]
            own[y * n + x] = 0 if o == 0 else (1 if o == mover else 2)
    return own


def play_game(evaluator, cfg, rng, opponent_ev=None):
    """Play one self-play game and return (examples, record).

    Exploration: the first `opening_random_plies` moves are uniformly random (and
    not recorded), so games start from diverse positions instead of collapsing
    onto one line.

    If `opponent_ev` is given, `evaluator` plays one (random) colour and the
    opponent net plays the other — and ONLY the main net's positions are recorded
    (on-policy targets for the net being trained), while the varied opponent
    still diversifies the trajectories."""
    n = cfg.board_size
    mcts = MCTS(evaluator, cfg)
    # distinct seed: MCTS seeds its Gumbel RNG from cfg.seed, so sharing cfg would
    # make the opponent draw the exact same noise sequence as the main net.
    mcts_opp = (MCTS(opponent_ev, replace(cfg, seed=cfg.seed + 991))
                if opponent_ev is not None else mcts)
    main_color = 1 if opponent_ev is None else (1 if rng.random() < 0.5 else 2)
    pure_selfplay = opponent_ev is None

    board = Board(n, n)
    positions = []            # (planes, pi, player) — main net's recorded positions
    moves = []
    reuse = None              # promoted subtree for tree reuse (pure self-play only)

    for ply in range(cfg.max_game_plies):
        if terminal_value(board) is not None:
            break
        mover = board.current_player

        if ply < cfg.opening_random_plies:            # random, unrecorded opening
            legal = board.possible_moves()
            x, y = legal[rng.integers(len(legal))]
            moves.append((x, y))
            board.move(x, y)
            reuse = None                              # tree is stale after a random move
            continue

        is_main = opponent_ev is None or mover == main_color
        # Playout Cap Randomization: record a full search on a fraction of the
        # main net's moves; play the rest cheaply and unrecorded.
        record = is_main and (rng.random() < cfg.pcr_prob)
        if is_main:
            budget = cfg.num_simulations if record else cfg.pcr_fast_sims
        else:
            budget = cfg.num_simulations
        r = reuse if (pure_selfplay and cfg.tree_reuse) else None
        # Gumbel noise IS the exploration device: on it for the first temp_moves
        # plies (varied selected action), off after (greedy). It replaces the old
        # tau=1 completed-policy sampling.
        explore = ply < cfg.temp_moves
        pi, root = (mcts if is_main else mcts_opp).search(
            board, add_noise=explore, max_sims=budget, reuse_root=r)
        if pi.sum() == 0:
            break

        if record:                                    # record only full-search main moves
            positions.append((encode(board), pi.astype(np.float32), mover))

        # Play the Sequential-Halving survivor, NOT an argmax of the completed
        # policy — the latter can pick an action that Gumbel search eliminated.
        # (The completed policy `pi` is still the recorded training target.)
        action = root.selected_action
        # ABLATION (self_play_gumbel=False): tau=1 visit-count sampling for the
        # first temp_moves plies — the pre-Gumbel exploration device.
        if not getattr(cfg, "self_play_gumbel", True) and explore:
            counts = root.child_N
            tot = counts.sum()
            if tot > 0:
                li = int(rng.choice(len(root.legal), p=counts / tot))
                action = int(root.legal[li])

        if pure_selfplay and cfg.tree_reuse:          # carry the chosen child forward
            li = np.where(root.legal == action)[0]
            reuse = root.children.get(int(li[0])) if len(li) else None
        else:
            reuse = None

        x, y = index_to_move(action, n)
        moves.append((x, y))
        board.move(x, y)

    winner = game_winner(board)
    examples = []
    for planes, pi, player in positions:
        z = 0.0 if winner == 0 else (1.0 if player == winner else -1.0)
        own = _ownership(board, player, n)
        examples.append((planes, pi, z, own))

    record = {"moves": moves, "size": n, "winner": winner, "plies": len(moves)}
    return examples, record


def expand_symmetries(examples, cfg):
    """Apply dihedral augmentation to a list of (planes, pi, z, own)."""
    if not cfg.augment_symmetries:
        return list(examples)
    out = []
    n = cfg.board_size
    for planes, pi, z, own in examples:
        for p2, pi2, own2 in augment(planes, pi, own, n, n):
            out.append((p2, pi2.astype(np.float32), z, own2))
    return out
