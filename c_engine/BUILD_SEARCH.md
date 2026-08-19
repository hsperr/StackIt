# Building the search engine

Build `stackit` from the real board implementation plus the search port:

```
clang -std=c99 -O3 -Wall -Wextra -o stackit board.c search.c main.c
```

Run it as:

```
./stackit <sx> <sy> "<moves>" <seconds> [max_depth]
```

`<moves>` is a space-separated list of `x,y` pairs applied from the initial
board (pass `""` for none). Prints one line of JSON, e.g.:

```
{"move":[2,2],"score":3,"depth":9,"nodes":2255961,"qnodes":242937,"secs":0.5000}
```

## Evaluation variants

`board.c` carries several `board_eval` implementations behind
`-DEVAL_VARIANT=n`. The default (4) is territory + chips + explosion threats,
which beat the old chip-count eval 63% at 0.05s/move and 70% at 0.3s/move over
60 and 30 games. See the comment block above `board_eval` for the full table.

Race them yourself — it builds its own binaries into a temp dir and never
touches `stackit`:

```
python c_engine/eval_tourney.py depth 40      # is the eval smarter?
python c_engine/eval_tourney.py time  60      # does it pay for the depth it costs?
```

**Never rebuild `c_engine/stackit` while a training run is live.** It is the
fixed-strength benchmark opponent; swapping it mid-run makes the AlphaBeta line
on the dashboard compare two different opponents. Rebuild between runs only, and
re-run `difftest.py` afterwards.
