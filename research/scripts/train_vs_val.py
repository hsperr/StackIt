"""Is the net memorising the training set at all?

'It cannot memorise' and 'it cannot generalise' are different failures with
different fixes. On 190 and 2000 positions the net memorises perfectly, so the
machinery works. This asks the same question on the FULL dataset: top-1 on data
it has trained on, versus held-out.

  train >> val  -> it memorises and fails to transfer: a generalisation problem,
                   and more capacity would only widen the gap
  train ~= val  -> it is not memorising either: the model class cannot express
                   the function, and the input or the architecture is at fault
"""
import sys
sys.path.insert(0, '/Users/hsperr/code/youtube_coding/StackIt')
sys.path.insert(0, '/tmp/stackit-handover-5/scripts')
import numpy as np
import torch
from alphazero.config import Config
from alphazero.net import StackNet
from alphazero.train import eval_net
from arch_bench import ConvPolicyNet, ArrayBuffer, load_data

ckpt, head = sys.argv[1], sys.argv[2]
data = sys.argv[3] if len(sys.argv) > 3 else '/tmp/stackit-handover-4/ab_data/ab3000.npz'
cfg = Config()
buf, vbuf = load_data(data, cfg.board_size)          # same seed 7 -> same split

d = torch.load(ckpt, map_location='cpu', weights_only=False)
a = d["arch"]
cls = ConvPolicyNet if head == "conv" else StackNet
net = cls(a["board_size"], a["channels"], a["res_blocks"])
net.load_state_dict(d["state"]); net.eval()

# equal-sized sample of the TRAINING data, so the two numbers are comparable
idx = np.random.default_rng(3).integers(0, len(buf), size=len(vbuf))
tbuf = ArrayBuffer(buf.P[idx], buf.PI[idx], buf.Z[idx], buf.OWN[idx])
dev = torch.device("cpu")
tp, tv, ttop1, tsign = eval_net(net, tbuf, cfg, dev)
vp, vv, vtop1, vsign = eval_net(net, vbuf, cfg, dev)
print(f"{ckpt}  ({a['channels']}x{a['res_blocks']}, head={head})")
print(f"  TRAIN (seen)  n={len(tbuf):>6}  policy CE {tp:.4f}  top1 {ttop1:.4f}  vsign {tsign:.4f}")
print(f"  VAL  (unseen) n={len(vbuf):>6}  policy CE {vp:.4f}  top1 {vtop1:.4f}  vsign {vsign:.4f}")
print(f"  gap: top1 {ttop1-vtop1:+.4f}")
