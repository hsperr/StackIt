"""Architecture bench: which change lifts the 0.46 top-1 ceiling?

Supervised imitation of AlphaBeta hits the same wall as self-play RL (0.46 vs
0.48) with train loss 1.499 against held-out 1.556 -- the net underfits, so the
limit is the network. This bench swaps one thing at a time against a FIXED
target, ~15 min an arm instead of hours of self-play.

Arms are (head, channels, blocks):
  fc   -- the shipped AlphaGo-Zero head: Conv2d(ch,2,1) then Linear(2*25,25).
          Everything the policy knows is squeezed through 2 channels, and the
          Linear shares no weights across cells, so each square's rule is learned
          separately. Prime suspect for both the ceiling and the measured 52%
          self-disagreement under board rotation.
  conv -- Leela/KataGo style: a 3x3 conv at full width, then a 1x1 conv to one
          logit per cell. Weight-shared across cells, no bottleneck.

Every arm gets the same data, the same schedule and the same stopping rule, so
the numbers are comparable. Stopping rule: stop the moment held-out top-1 has
not improved by 0.002 for `patience` rounds. No LR annealing -- a measured
LR drop bought the baseline 0.450 -> 0.467 and then flattened, and squeezing out
another point is not the question. We are screening for a change that moves the
curve's SHAPE, so an arm that is merely fine-tuning is an arm that has failed.
"""
import sys, time, json, os, glob
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero.config import Config
from alphazero.net import StackNet
from alphazero.encoding import augment
from alphazero.train import train_net, eval_net
from alphazero import metrics


class ConvPolicyNet(StackNet):
    """StackNet with the 2-channel + Linear policy head replaced by a per-cell
    convolutional one. Value and ownership heads are untouched."""

    def __init__(self, board_size, channels=64, res_blocks=4, phead=32):
        super().__init__(board_size, channels, res_blocks)
        self.p_conv = nn.Conv2d(channels, phead, 3, padding=1, bias=False)
        self.p_bn = nn.BatchNorm2d(phead)
        self.p_fc = None                       # drop the fully-connected layer
        self.p_out = nn.Conv2d(phead, 1, 1)    # one logit per cell

    def forward(self, x):
        h = self.tower(self.stem(x))
        p = F.relu(self.p_bn(self.p_conv(h)))
        p = self.p_out(p).flatten(1)
        v = F.relu(self.v_bn(self.v_conv(h)))
        v = F.relu(self.v_fc1(v.flatten(1)))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)
        o = F.relu(self.o_bn(self.o_conv1(h)))
        o = self.o_conv2(o)
        return p, v, o


class _View:
    """Row accessor so eval_net's `buf[j]` keeps working. Planes are stored
    uint8 to keep 1.7M positions under a GB; widen here, because eval_net hands
    the row straight to torch and MPS will not mix a Byte input with Float
    weights."""

    def __init__(self, b):
        self.b = b

    def __getitem__(self, i):
        return (self.b.P[i].astype(np.float32), self.b.PI[i],
                self.b.Z[i], self.b.OWN[i])


class ArrayBuffer:
    """Same interface as alphazero.replay.ReplayBuffer, but array-backed: that
    one is a deque, and random-indexing 2M elements is O(n) per sample.

    With `augment_live=True` a random dihedral op is drawn per batch and applied
    to planes, policy and ownership together. That replaces materialising all 8
    copies -- at 2M positions those would need ~19 GB -- and it is strictly better
    than a fixed 8x expansion, since every epoch redraws the orientation instead
    of revisiting the same eight rows. One op per batch, not per row: the ops are
    array-wide numpy calls, and over thousands of steps the orientation mix is the
    same. Matches encoding._apply_planes / _apply_grid exactly.
    """

    def __init__(self, P, PI, Z, OWN, augment_live=False, board=5):
        self.P, self.PI, self.Z, self.OWN = P, PI, Z, OWN
        self.augment_live, self.board = augment_live, board
        self.buf = _View(self)

    def __len__(self):
        return len(self.P)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, len(self.P), size=min(batch_size, len(self.P)))
        p, pi, own = self.P[idx].astype(np.float32), self.PI[idx], self.OWN[idx]
        if self.augment_live:
            n = self.board
            k = np.random.randint(4)
            flip = np.random.randint(2) == 1
            pi = pi.reshape(-1, n, n)
            own = own.reshape(-1, n, n)
            if k:
                p = np.rot90(p, k=k, axes=(2, 3))
                pi = np.rot90(pi, k=k, axes=(1, 2))
                own = np.rot90(own, k=k, axes=(1, 2))
            if flip:
                p = p[:, :, :, ::-1]
                pi = pi[:, :, ::-1]
                own = own[:, :, ::-1]
            p = np.ascontiguousarray(p)
            pi = np.ascontiguousarray(pi.reshape(-1, n * n))
            own = np.ascontiguousarray(own.reshape(-1, n * n))
        return (torch.from_numpy(p).to(device),
                torch.from_numpy(pi).to(device),
                torch.from_numpy(self.Z[idx]).to(device),
                torch.from_numpy(own).to(device))


def load_data(path, board, val_frac=0.05, do_augment=True, live=False):
    """`path` may be a glob over several shards."""
    files = sorted(glob.glob(path)) or [path]
    # Preallocate and fill shard by shard: np.concatenate over six shards holds
    # both the parts and the result at once, and that peak is what got a run
    # SIGKILLed. Planes go to uint8 (lossless, they are binary) -- 4x smaller.
    sizes, n = [], 0
    for f in files:
        with np.load(f) as q:
            sizes.append(len(q["z"])); n += sizes[-1]
    P = np.empty((n, 9, board, board), np.uint8)
    PI = np.empty((n, board * board), np.float32)
    Z = np.empty(n, np.float32)
    OWN = np.empty((n, board * board), np.int64)
    o = 0
    for f, m in zip(files, sizes):
        with np.load(f) as q:
            P[o:o + m] = q["planes"].astype(np.uint8)
            PI[o:o + m] = q["pi"]
            Z[o:o + m] = q["z"]
            OWN[o:o + m] = q["own"]
        o += m
    print(f"  loaded {len(files)} file(s), {n:,} positions, "
          f"{(P.nbytes + PI.nbytes + Z.nbytes + OWN.nbytes)/1e9:.2f} GB resident",
          flush=True)
    perm = np.random.default_rng(7).permutation(len(P))
    nv = int(len(P) * val_frac)
    v, t = perm[:nv], perm[nv:]
    vbuf = ArrayBuffer(P[v], PI[v], Z[v], OWN[v])       # held out BEFORE augmenting
    if live:
        # augment per batch instead of materialising 8 copies
        return ArrayBuffer(P[t], PI[t], Z[t], OWN[t], augment_live=True,
                           board=board), vbuf
    if not do_augment:
        # Canonicalised data: every board is already in one fixed orientation, so
        # augmenting would reintroduce the 7 orientations the net no longer has to
        # model -- and would put val positions' symmetries into the training set.
        return ArrayBuffer(P[t], PI[t], Z[t], OWN[t]), vbuf
    oP, oPI, oZ, oOWN = [], [], [], []
    for lo in range(0, len(t), 20000):
        idx = t[lo:lo + 20000]
        cP, cPI, cZ, cOWN = [], [], [], []
        for i in idx:
            for p2, pi2, own2 in augment(P[i], PI[i], OWN[i], board, board):
                cP.append(p2); cPI.append(pi2.astype(np.float32))
                cZ.append(Z[i]); cOWN.append(own2)
        oP.append(np.asarray(cP, np.float32)); oPI.append(np.asarray(cPI, np.float32))
        oZ.append(np.asarray(cZ, np.float32)); oOWN.append(np.asarray(cOWN, np.int64))
    buf = ArrayBuffer(np.concatenate(oP), np.concatenate(oPI),
                      np.concatenate(oZ), np.concatenate(oOWN))
    return buf, vbuf


def run_arm(name, head, channels, blocks, buf, vbuf, out_root,
            steps=400, max_rounds=400, patience=15, min_delta=0.002, max_drops=0):
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    cfg = Config(ckpt_dir=out, metrics_file=os.path.join(out, "metrics.jsonl"),
                 channels=channels, res_blocks=blocks)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    cls = ConvPolicyNet if head == "conv" else StackNet
    net = cls(cfg.board_size, channels, blocks).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"\n=== arm {name}: head={head} {channels}x{blocks} "
          f"{nparam/1e3:.0f}k params ===", flush=True)

    # A fixed, val-sized sample of the TRAINING data. The running train loss from
    # train_net is an average over a changing net mid-epoch; this is the same
    # measurement as the held-out one, so the two numbers are directly comparable
    # and the fit/generalise split is visible every round.
    ti = np.random.default_rng(3).integers(0, len(buf), size=len(vbuf))
    tbuf = ArrayBuffer(buf.P[ti], buf.PI[ti], buf.Z[ti], buf.OWN[ti])

    mf = open(os.path.join(out, "metrics.jsonl"), "w")
    epoch = len(buf) / (steps * cfg.batch_size)
    print(f"  {len(buf):,} train rows: 1 epoch = {epoch:.1f} rounds; "
          f"max_rounds={max_rounds} = {max_rounds/epoch:.1f} epochs", flush=True)
    # `best` tracks the high-water mark and updates on ANY gain, so the reported
    # number is the best score actually reached. `ref` is the separate value the
    # patience counter measures min_delta against -- the old code folded the two
    # together, so a steady climb of <min_delta per round never reset the counter
    # and the arm was killed while still improving (fc64x4 hit 0.4567 at r90 and
    # reported 0.4553).
    best, ref, best_r, drops, t_arm = 0.0, 0.0, 0, 0, time.time()
    for r in range(1, max_rounds + 1):
        t0 = time.time()
        p, vl, o = train_net(net, opt, buf, cfg, dev, steps)
        tp, _, ttop1, tsign = eval_net(net, tbuf, cfg, dev)
        vp, vv, top1, sign = eval_net(net, vbuf, cfg, dev)
        lr = opt.param_groups[0]["lr"]
        mf.write(json.dumps({"round": r, "lr": lr, "policy_loss": round(p, 4),
                             "train_policy_loss": round(tp, 4),
                             "train_top1": round(ttop1, 4),
                             "val_policy_loss": round(vp, 4), "policy_top1": round(top1, 4),
                             "value_sign_acc": round(sign, 4),
                             "sec": round(time.time() - t0, 1)}) + "\n")
        mf.flush()
        if r % 10 == 0 or r <= 3:
            print(f"  r{r:>3} lr={lr:.1e} | train loss {tp:.3f} top1 {ttop1:.4f} "
                  f"| test loss {vp:.3f} top1 {top1:.4f} | gap {ttop1-top1:+.4f}",
                  flush=True)
        if top1 > best:
            best = top1
            metrics.save_checkpoint(cfg, net, "best.pt",
                                    extra={"round": r, "top1": round(top1, 4),
                                           "head": head, "channels": channels,
                                           "blocks": blocks})
        if top1 > ref + min_delta:
            ref, best_r = top1, r
        elif r - best_r >= patience:
            if drops >= max_drops:
                print(f"  converged at r{r} (no gain for {patience} rounds "
                      f"after {drops} LR drops)", flush=True)
                break
            drops += 1
            for g in opt.param_groups:
                g["lr"] = g["lr"] / 3.0
            best_r = r
            print(f"  LR drop {drops} -> {opt.param_groups[0]['lr']:.1e} at r{r}",
                  flush=True)
    mf.close()
    print(f"=== arm {name}: best top-1 {best:.4f}  ({time.time()-t_arm:.0f}s) ===",
          flush=True)
    return {"arm": name, "head": head, "channels": channels, "blocks": blocks,
            "params": nparam, "best_top1": round(best, 4)}


def main():
    data, out_root = sys.argv[1], sys.argv[2]
    arms = [a.split(":") for a in sys.argv[3].split(",")]   # head:channels:blocks
    # argv[4]: 1 = pre-expand 8x (default), 0 = no augmentation, "live" = per-batch
    mode = sys.argv[4] if len(sys.argv) > 4 else "1"
    # argv[5] max_rounds, argv[6] patience. Both must scale with the dataset:
    # a round is a fixed 400x128 rows, so on 1.65M rows an epoch is 32 rounds and
    # the default patience of 15 declares convergence after half an epoch.
    max_rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 400
    patience = int(sys.argv[6]) if len(sys.argv) > 6 else 15
    cfg = Config()
    buf, vbuf = load_data(data, cfg.board_size, do_augment=(mode != "0"),
                          live=(mode == "live"))
    print(f"train {len(buf)} augmented, val {len(vbuf)}", flush=True)
    res = []
    for head, ch, bl in arms:
        res.append(run_arm(f"{head}{ch}x{bl}", head, int(ch), int(bl),
                           buf, vbuf, out_root,
                           max_rounds=max_rounds, patience=patience))
        with open(os.path.join(out_root, "summary.json"), "w") as f:
            json.dump(res, f, indent=2)
    print("\n=== summary ===", flush=True)
    for r in sorted(res, key=lambda r: -r["best_top1"]):
        print(f"  {r['arm']:<12} {r['params']/1e3:>7.0f}k params  "
              f"top-1 {r['best_top1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
