"""Can the net perfectly fit a handful of positions?

The capacity story only holds if the optimiser can drive TRAINING error to zero
on a tiny set. If it cannot memorise ~2 games, the fault is not capacity or
generalisation -- it is a bug in the encoding, the targets, or the loss, and
every other measurement is built on sand.

No augmentation, no weight decay, no held-out split: we deliberately want
overfitting. Score is top-1 on the very data being trained on, so anything short
of ~1.00 is a red flag.

Runs on CPU on purpose -- it is tiny, and the MPS device is busy with the arch
bench.
"""
import sys, json
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero.config import Config
from alphazero.net import StackNet
from alphazero.train import train_net, eval_net
sys.path.insert(0, '/tmp/stackit-handover-4/scripts')
from arch_bench import ConvPolicyNet, ArrayBuffer


def main():
    data = sys.argv[1]
    n_pos = int(sys.argv[2]) if len(sys.argv) > 2 else 190      # ~2 games
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    d = np.load(data)
    P = d["planes"][:n_pos].astype(np.float32)
    PI = d["pi"][:n_pos].astype(np.float32)
    Z = d["z"][:n_pos].astype(np.float32)
    OWN = d["own"][:n_pos].astype(np.int64)
    buf = ArrayBuffer(P, PI, Z, OWN)

    # Sanity on the data itself: identical encodings carrying different targets
    # are unfittable by ANY network, and would explain a sub-1.0 ceiling.
    keys = {}
    clash = 0
    for i in range(len(P)):
        k = P[i].tobytes()
        a = int(PI[i].argmax())
        if k in keys and keys[k] != a:
            clash += 1
        keys[k] = a
    print(f"{len(P)} positions, {len(keys)} distinct encodings, "
          f"{clash} contradictory duplicate labels", flush=True)

    cfg = Config(ckpt_dir="/tmp/stackit-handover-4/memo", weight_decay=0.0)
    dev = torch.device("cpu")
    for head in ("fc", "conv"):
        torch.manual_seed(0)
        cls = ConvPolicyNet if head == "conv" else StackNet
        net = cls(cfg.board_size, 64, 4).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0)
        print(f"\n--- head={head} (64x4), fitting {len(P)} positions ---", flush=True)
        for r in range(1, rounds + 1):
            p, v, o = train_net(net, opt, buf, cfg, dev, steps)
            tp, tv, top1, sign = eval_net(net, buf, cfg, dev)   # train data on purpose
            if r % 5 == 0 or r <= 3:
                print(f"  r{r:>3} train p={p:.4f} | fit p={tp:.4f} "
                      f"top1={top1:.4f} vsign={sign:.4f}", flush=True)
            if top1 >= 0.999:
                print(f"  MEMORISED at round {r}", flush=True)
                break
        else:
            print(f"  did NOT memorise in {rounds} rounds (top1={top1:.4f})", flush=True)


if __name__ == "__main__":
    main()
