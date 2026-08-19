"""Compare two training runs round-by-round on the metrics that matter.

Elo is measured on 4-game gauntlets and swings +-100 for free, so it cannot
settle an experiment. These four can:
  * val_policy_loss  — held-out fit to the search target (the headline number)
  * policy_top1      — how often the net's best move IS the search's best move
  * value_sign_acc   — how often the value head calls the winner (0.5 = coin flip)
  * gap              — train minus held-out policy loss. Positive and growing
                       means overfitting: the arm is memorising the buffer.

Run:  python3 -m alphazero.compare_runs checkpoints checkpoints_1200
"""
import os
import sys
import json


def load(d):
    p = os.path.join(d, "metrics.jsonl")
    with open(p) as f:
        return {r["iter"]: r for r in (json.loads(l) for l in f if l.strip())}


def main():
    a_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints"
    b_dir = sys.argv[2] if len(sys.argv) > 2 else "checkpoints_1200"
    A, B = load(a_dir), load(b_dir)
    shared = sorted(i for i in set(A) & set(B) if i > 24)
    if not shared:
        print(f"no shared rounds past 24 yet ({a_dir}: max {max(A)}, {b_dir}: max {max(B)})")
        return
    print(f"A = {a_dir}   B = {b_dir}")
    print(f"{'it':>3} | {'valP A':>7} {'B':>7} | {'top1 A':>7} {'B':>6} | "
          f"{'vsign A':>7} {'B':>6} | {'gap A':>6} {'B':>6}")
    for i in shared:
        a, b = A[i], B[i]
        ga = a["policy_loss"] - a["val_policy_loss"]
        gb = b["policy_loss"] - b["val_policy_loss"]
        print(f"{i:>3} | {a['val_policy_loss']:>7.3f} {b['val_policy_loss']:>7.3f} | "
              f"{a['policy_top1']:>7.3f} {b['policy_top1']:>6.3f} | "
              f"{a['value_sign_acc']:>7.3f} {b['value_sign_acc']:>6.3f} | "
              f"{ga:>6.3f} {gb:>6.3f}")
    last = shared[-5:]
    def avg(D, k):
        return sum(D[i][k] for i in last) / len(last)
    print(f"\nmean over rounds {last[0]}-{last[-1]}:")
    for k, name in (("val_policy_loss", "held-out policy (lower better)"),
                    ("policy_top1", "top1            (higher better)"),
                    ("value_sign_acc", "value sign acc  (higher better)")):
        va, vb = avg(A, k), avg(B, k)
        who = "B" if ((vb < va) == (k == "val_policy_loss")) else "A"
        print(f"  {name}: A {va:.4f}   B {vb:.4f}   -> {who} wins")


if __name__ == "__main__":
    main()
