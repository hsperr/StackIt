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
