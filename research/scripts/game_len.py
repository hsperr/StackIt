"""How long do self-play games run with Gumbel noise on EVERY ply vs capped?

Root Gumbel noise is the exploration device, but a noisy move late in StackIt can
stop a game from ever reaching domination. If games hit `max_game_plies` the
winner is decided by box count — a different rule — which poisons the value label.
"""
import sys, time
import numpy as np

sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')

from alphazero.config import Config
from alphazero.net import StackNet
from alphazero.net import Evaluator
from alphazero.selfplay import play_game
from alphazero.metrics import load_net

GAMES = 6


def run(cfg, ev, tag):
    rng = np.random.default_rng(11)
    lens, caps = [], 0
    t = time.time()
    for i in range(GAMES):
        _, rec = play_game(ev, cfg, rng)
        lens.append(rec['plies'])
        caps += int(rec['plies'] >= cfg.max_game_plies)
    l = np.array(lens)
    print(f"{tag:>28}: median {np.median(l):5.0f} plies  max {l.max():4d}  "
          f"hit-cap {caps}/{GAMES}  {(time.time()-t)/GAMES:.1f}s/game", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "random"
    if which == "random":
        net = StackNet(5, 64, 4); net.eval()
    else:
        net, _ = load_net(which, 'cpu')
    ev = Evaluator(net, 'cpu')
    for npl in (0, 16):
        cfg = Config(channels=64, res_blocks=4, num_simulations=256,
                     gumbel_noise_plies=npl)
        run(cfg, ev, f"noise_plies={npl or 'all'}")


if __name__ == "__main__":
    main()
