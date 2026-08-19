#!/bin/zsh
# Generate teacher games as independent shards so the run can be stopped at any
# point and everything finished so far is still usable. Each shard gets its own
# seed block; overlapping seeds would replay the same games and add nothing.
DATA=/tmp/stackit-handover-4/ab_data
for i in 1 2 3 4 5 6; do
  OUT=$DATA/ab_shard$i.npz
  [[ -f $OUT ]] && { echo "skip shard$i (exists)"; continue; }
  SEED=$((2000000 + i * 500000))
  echo "=== shard $i  seed0=$SEED  $(date +%H:%M) ==="
  nice -n 5 python /tmp/stackit-handover-5/scripts/gen_ab_games.py 3000 10 0.05 $OUT $SEED
done
echo "=== all shards done $(date +%H:%M) ==="
