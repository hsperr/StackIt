"""Supervised pretraining: can the net learn to imitate AlphaBeta at all?

Self-play RL stalled at top-1 0.48, and the AlphaBeta bar it was measured
against turned out to be unreproducible, so we cannot tell whether the *net* is
the limit. Imitation is a fixed target with no moving parts: feed it AlphaBeta's
chosen move and see how far top-1 goes.

Reads:  top-1 stalls near 0.48 -> the ARCHITECTURE is the ceiling (cf. the 52%
                                  self-disagreement under board rotation)
        top-1 climbs well past -> the net is fine, the RL loop is the problem

Reuses alphazero's own train_net/eval_net/augment so the loss, the augmentation
and the held-out metrics mean exactly what they mean in the real trainer. The
checkpoint is loadable by alphazero.metrics.load_net, so the result can seed
self-play directly.

Storage is array-backed, NOT alphazero.replay.ReplayBuffer: that is a deque, and
random indexing a 2M-element deque is O(n) per sample. Same .sample()/.buf
interface, so train_net and eval_net do not know the difference.
"""
import sys, time, glob, json, os
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
import numpy as np
import torch

from alphazero.config import Config
from alphazero.net import StackNet
from alphazero.encoding import augment
from alphazero.train import train_net, eval_net
from alphazero import metrics


class _View:
    """Row accessor so eval_net's `val_buffer.buf[j]` keeps working."""
    def __init__(self, b):
        self.b = b

    def __getitem__(self, i):
        return (self.b.P[i], self.b.PI[i], self.b.Z[i], self.b.OWN[i])


class ArrayBuffer:
    def __init__(self, P, PI, Z, OWN):
        self.P, self.PI, self.Z, self.OWN = P, PI, Z, OWN
        self.buf = _View(self)

    def __len__(self):
        return len(self.P)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, len(self.P), size=min(batch_size, len(self.P)))
        return (torch.from_numpy(self.P[idx]).to(device),
                torch.from_numpy(self.PI[idx]).to(device),
                torch.from_numpy(self.Z[idx]).to(device),
                torch.from_numpy(self.OWN[idx]).to(device))


def augment_arrays(P, PI, Z, OWN, n):
    """Dihedral expansion straight into contiguous arrays, in chunks so peak
    memory stays near the final size instead of 8x it in Python objects."""
    outP, outPI, outZ, outOWN = [], [], [], []
    for lo in range(0, len(P), 20000):
        hi = min(lo + 20000, len(P))
        cP, cPI, cZ, cOWN = [], [], [], []
        for i in range(lo, hi):
            for p2, pi2, own2 in augment(P[i], PI[i], OWN[i], n, n):
                cP.append(p2); cPI.append(pi2.astype(np.float32))
                cZ.append(Z[i]); cOWN.append(own2)
        outP.append(np.asarray(cP, dtype=np.float32))
        outPI.append(np.asarray(cPI, dtype=np.float32))
        outZ.append(np.asarray(cZ, dtype=np.float32))
        outOWN.append(np.asarray(cOWN, dtype=np.int64))
        print(f"  augmented {hi}/{len(P)}", flush=True)
    return (np.concatenate(outP), np.concatenate(outPI),
            np.concatenate(outZ), np.concatenate(outOWN))


def main():
    pattern, out_dir = sys.argv[1], sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    channels = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    blocks = int(sys.argv[6]) if len(sys.argv) > 6 else 4

    os.makedirs(out_dir, exist_ok=True)
    cfg = Config(ckpt_dir=out_dir, metrics_file=os.path.join(out_dir, "metrics.jsonl"),
                 channels=channels, res_blocks=blocks)

    P = PI = Z = OWN = None
    for f in sorted(glob.glob(pattern)):
        d = np.load(f)
        a, b, c, e = (d["planes"].astype(np.float32), d["pi"].astype(np.float32),
                      d["z"].astype(np.float32), d["own"].astype(np.int64))
        P = a if P is None else np.concatenate([P, a])
        PI = b if PI is None else np.concatenate([PI, b])
        Z = c if Z is None else np.concatenate([Z, c])
        OWN = e if OWN is None else np.concatenate([OWN, e])
        print(f"  loaded {f}: {len(a)} positions", flush=True)

    rng = np.random.default_rng(7)
    perm = rng.permutation(len(P))
    n_val = int(len(P) * 0.05)
    v, t = perm[:n_val], perm[n_val:]
    # hold out BEFORE augmentation so no symmetry of a val position is trained on
    vbuf = ArrayBuffer(P[v], PI[v], Z[v], OWN[v])
    aP, aPI, aZ, aOWN = augment_arrays(P[t], PI[t], Z[t], OWN[t], cfg.board_size)
    del P, PI, Z, OWN
    buf = ArrayBuffer(aP, aPI, aZ, aOWN)
    print(f"train {len(t)} -> {len(buf)} augmented, val {len(vbuf)}, "
          f"{aP.nbytes/1e9:.1f} GB planes", flush=True)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = StackNet(cfg.board_size, channels, blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mf = open(os.path.join(out_dir, "metrics.jsonl"), "w")
    best = 0.0
    for r in range(1, rounds + 1):
        t0 = time.time()
        p, val_, o = train_net(net, opt, buf, cfg, dev, steps)
        vp, vv, top1, sign = eval_net(net, vbuf, cfg, dev)
        row = {"round": r, "policy_loss": round(p, 4), "value_loss": round(val_, 4),
               "own_loss": round(o, 4), "val_policy_loss": round(vp, 4),
               "val_value_loss": round(vv, 4), "policy_top1": round(top1, 4),
               "value_sign_acc": round(sign, 4), "sec": round(time.time() - t0, 1)}
        mf.write(json.dumps(row) + "\n"); mf.flush()
        print(f"r{r:>3} train p={p:.3f} | val p={vp:.3f} top1={top1:.4f} "
              f"vsign={sign:.4f} ({row['sec']}s)", flush=True)
        if top1 > best:
            best = top1
            metrics.save_checkpoint(cfg, net, "best.pt",
                                    extra={"round": r, "top1": round(top1, 4)})
    mf.close()
    print(f"best held-out top-1 agreement with AlphaBeta: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
