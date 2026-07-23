"""Checkpoint + metrics I/O. The training loop writes one JSON line per
iteration to metrics.jsonl and dumps the latest self-play game to last_game.json
— both consumed live by the dashboard.
"""
import os
import json

import torch

from .net import StackNet


def ensure_dir(cfg):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)


def append_metric(cfg, row):
    ensure_dir(cfg)
    with open(cfg.metrics_file, "a") as f:
        f.write(json.dumps(row) + "\n")


def write_last_game(cfg, record):
    ensure_dir(cfg)
    with open(os.path.join(cfg.ckpt_dir, "last_game.json"), "w") as f:
        json.dump(record, f)


def write_status(cfg, status):
    ensure_dir(cfg)
    with open(os.path.join(cfg.ckpt_dir, "status.json"), "w") as f:
        json.dump(status, f)


def write_ratings(cfg, ratings):
    ensure_dir(cfg)
    with open(os.path.join(cfg.ckpt_dir, "ratings.json"), "w") as f:
        json.dump(ratings, f)


def save_version(cfg, net, vid):
    d = os.path.join(cfg.ckpt_dir, "versions")
    os.makedirs(d, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save({"arch": net.arch(), "state": state}, os.path.join(d, f"v{vid}.pt"))


def save_champion(cfg, arch, state, champion_id, elo):
    """Write best.pt from an arbitrary champion's arch+state (may be a reference
    of a different architecture than the net currently training)."""
    ensure_dir(cfg)
    torch.save({"arch": arch, "state": state, "cfg": cfg.to_dict(),
                "extra": {"version": champion_id, "elo": elo}},
               os.path.join(cfg.ckpt_dir, "best.pt"))


def load_version_state(cfg, vid):
    path = os.path.join(cfg.ckpt_dir, "versions", f"v{vid}.pt")
    return torch.load(path, map_location="cpu", weights_only=False)["state"]


def save_checkpoint(cfg, net, name, extra=None):
    ensure_dir(cfg)
    path = os.path.join(cfg.ckpt_dir, name)
    torch.save({"arch": net.arch(), "state": net.state_dict(),
                "cfg": cfg.to_dict(), "extra": extra or {}}, path)
    return path


def load_net(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = ckpt["arch"]
    net = StackNet(a["board_size"], a["channels"], a["res_blocks"]).to(device)
    net.load_state_dict(ckpt["state"])
    net.eval()
    return net, ckpt
