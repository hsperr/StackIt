"""At what dataset size does the net stop being able to MEMORISE?

Everything so far early-stopped on validation after ~2 epochs, so nothing was
ever given the steps to fit its training set. That makes 'the net cannot fit
268k positions' unproven, and it makes '5.7x the parameters changes nothing'
weaker than it looked -- both sizes ran the same 2 epochs.

This removes validation from the stopping rule entirely and trains each subset
until TRAIN top-1 stops moving. The size where memorisation breaks separates the
two remaining stories:

  breaks well below 268k -> a capacity/optimisation wall; the 0.46 val ceiling is
                            downstream of simply not being able to fit
  memorises all the way  -> capacity is genuinely fine, the net just cannot
                            GENERALISE, and the input encoding is the suspect

No augmentation: we are asking about fitting these exact rows. Held-out top-1 is
reported alongside purely as context -- it is never used to stop.
"""
import sys, time, json, os
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
sys.path.insert(0, '/tmp/stackit-handover-5/scripts')
import numpy as np
import torch

from alphazero.config import Config
from alphazero.net import StackNet
from alphazero.train import train_net, eval_net
from arch_bench import ConvPolicyNet, ArrayBuffer


def main():
    data = sys.argv[1]
    out_root = sys.argv[2]
    sizes = [int(x) for x in sys.argv[3].split(",")]
    head = sys.argv[4] if len(sys.argv) > 4 else "conv"
    channels = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    blocks = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    steps, max_rounds, patience = 400, 400, 25

    os.makedirs(out_root, exist_ok=True)
    d = np.load(data)
    P, PI = d["planes"].astype(np.float32), d["pi"].astype(np.float32)
    Z, OWN = d["z"].astype(np.float32), d["own"].astype(np.int64)
    perm = np.random.default_rng(7).permutation(len(P))       # same split as the bench
    nv = int(len(P) * 0.05)
    v, t = perm[:nv], perm[nv:]
    vbuf = ArrayBuffer(P[v], PI[v], Z[v], OWN[v])
    cfg = Config()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    res = []

    for n in sizes:
        idx = t[:n]
        buf = ArrayBuffer(P[idx], PI[idx], Z[idx], OWN[idx])
        torch.manual_seed(0)
        cls = ConvPolicyNet if head == "conv" else StackNet
        net = cls(cfg.board_size, channels, blocks).to(dev)
        nparam = sum(q.numel() for q in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        per_round = steps * cfg.batch_size / n
        print(f"\n=== n={n} ({head} {channels}x{blocks}, {nparam/1e3:.0f}k params, "
              f"{per_round:.2f} epochs/round) ===", flush=True)
        mf = open(os.path.join(out_root, f"n{n}.jsonl"), "w")
        best_tr, best_r, t0 = 0.0, 0, time.time()
        for r in range(1, max_rounds + 1):
            train_net(net, opt, buf, cfg, dev, steps)
            tp, _, ttop1, tsign = eval_net(net, buf, cfg, dev)     # fit on seen data
            vp, _, vtop1, vsign = eval_net(net, vbuf, cfg, dev)    # context only
            mf.write(json.dumps({"round": r, "n": n, "train_policy_loss": round(tp, 4),
                                 "train_top1": round(ttop1, 4),
                                 "val_policy_loss": round(vp, 4),
                                 "val_top1": round(vtop1, 4),
                                 "epochs": round(r * per_round, 2)}) + "\n")
            mf.flush()
            if r % 10 == 0 or r <= 3:
                print(f"  r{r:>3} ep{r*per_round:>6.1f} | train loss {tp:.4f} "
                      f"top1 {ttop1:.4f} | test loss {vp:.4f} top1 {vtop1:.4f}",
                      flush=True)
            if ttop1 > best_tr + 0.002:
                best_tr, best_r = ttop1, r
            if ttop1 >= 0.999:
                print(f"  MEMORISED at r{r} ({r*per_round:.1f} epochs)", flush=True)
                break
            if r - best_r >= patience:
                print(f"  train top-1 stalled at {best_tr:.4f} "
                      f"({r*per_round:.1f} epochs)", flush=True)
                break
        mf.close()
        res.append({"n": n, "best_train_top1": round(best_tr, 4),
                    "final_val_top1": round(vtop1, 4), "rounds": r,
                    "epochs": round(r * per_round, 1), "sec": round(time.time() - t0)})
        with open(os.path.join(out_root, "summary.json"), "w") as f:
            json.dump(res, f, indent=2)

    print("\n=== summary: dataset size vs ability to memorise ===", flush=True)
    print(f"{'n':>8} {'train top1':>11} {'test top1':>10} {'epochs':>8}", flush=True)
    for r in res:
        print(f"{r['n']:>8} {r['best_train_top1']:>11.4f} "
              f"{r['final_val_top1']:>10.4f} {r['epochs']:>8.1f}", flush=True)


if __name__ == "__main__":
    main()
