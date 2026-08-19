"""Ship the run's OWN saved cfg through the same spawn-pool round trip the
gauntlet uses, and ask the worker which AlphaBeta class it ends up with."""
import sys
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
import os, torch
from multiprocessing import get_context
from alphazero.config import Config
from alphazero.arena_eval import make_alphabeta

def probe(cfg_dict):
    cfg = Config(**cfg_dict)
    p = make_alphabeta(cfg, 0.05)
    return (type(p).__name__, getattr(cfg, "alphabeta_engine", "<missing>"),
            getattr(p, "binary", None), os.path.exists(getattr(p, "binary", "")) )

if __name__ == "__main__":
    ck = torch.load('checkpoints_cover/best.pt', map_location='cpu', weights_only=False)
    d = ck['cfg']
    print("saved cfg alphabeta_engine =", d.get('alphabeta_engine'))
    print("round-trip keeps it        =", Config(**d).to_dict().get('alphabeta_engine'))
    with get_context("spawn").Pool(1) as pool:
        print("worker builds              =", pool.apply(probe, (d,)))
