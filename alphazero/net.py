"""Residual network with three heads over a shared body: policy, value, and a
per-cell ownership head (final owner of each cell).

Small by design (a few residual blocks, 64 filters) so self-play — which calls
this once per MCTS simulation — stays fast on CPU/MPS. BatchNorm is used for
training stability; MCTS inference must run under `model.eval()` (running stats)
so batch-1 forwards behave.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import NUM_PLANES, encode, legal_mask


def resolve_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(x + h)


class StackNet(nn.Module):
    def __init__(self, board_size, channels=64, res_blocks=4):
        super().__init__()
        self.board_size = board_size
        n_actions = board_size * board_size

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(res_blocks)])

        # policy head
        self.p_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * n_actions, n_actions)

        # value head
        self.v_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(n_actions, channels)
        self.v_fc2 = nn.Linear(channels, 1)

        # ownership head: per-cell 3-way logits (empty / mine / theirs at game end)
        self.o_conv1 = nn.Conv2d(channels, 32, 3, padding=1, bias=False)
        self.o_bn = nn.BatchNorm2d(32)
        self.o_conv2 = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        h = self.tower(self.stem(x))

        p = F.relu(self.p_bn(self.p_conv(h)))
        p = self.p_fc(p.flatten(1))                       # policy logits

        v = F.relu(self.v_bn(self.v_conv(h)))
        v = F.relu(self.v_fc1(v.flatten(1)))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)         # value in [-1, 1]

        o = F.relu(self.o_bn(self.o_conv1(h)))
        o = self.o_conv2(o)                               # [B, 3, N, N] ownership logits
        return p, v, o

    # ---- convenience config for checkpointing ----
    def arch(self):
        return {"board_size": self.board_size,
                "channels": self.stem[0].out_channels,
                "res_blocks": len(self.tower)}


class Evaluator:
    """Wraps a net for MCTS: encodes a Board, runs a masked-softmax forward, and
    returns (priors over legal actions, value) from the current player's view."""

    def __init__(self, net, device):
        self.net = net
        self.device = device

    @torch.no_grad()
    def infer(self, board):
        self.net.eval()
        planes = encode(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        logits, value, _ = self.net(x)                    # ownership head unused at inference
        logits = logits[0].float().cpu().numpy()
        v = float(value[0].item())

        mask = legal_mask(board)
        logits = np.where(mask, logits, -1e9)
        logits -= logits.max()
        exp = np.exp(logits)
        exp *= mask                       # zero illegal
        s = exp.sum()
        priors = exp / s if s > 0 else mask / max(mask.sum(), 1)
        return priors.astype(np.float32), v

    @torch.no_grad()
    def infer_batch(self, boards):
        """Batched version for many leaf boards at once (used by parallel eval)."""
        self.net.eval()
        x = torch.from_numpy(np.stack([encode(b) for b in boards])).to(self.device)
        logits, values, _ = self.net(x)                   # ownership head unused at inference
        logits = logits.float().cpu().numpy()
        values = values.float().cpu().numpy()
        out = []
        for i, b in enumerate(boards):
            mask = legal_mask(b)
            lg = np.where(mask, logits[i], -1e9)
            lg -= lg.max()
            exp = np.exp(lg) * mask
            s = exp.sum()
            priors = exp / s if s > 0 else mask / max(mask.sum(), 1)
            out.append((priors.astype(np.float32), float(values[i])))
        return out
