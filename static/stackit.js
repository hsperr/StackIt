/* ==========================================================================
   StackIt — front-end engine
   - vanilla fetch (no jQuery/Bootstrap)
   - swaps board.html partial into #board
   - client-side topple / chain-reaction animation by diffing board snapshots
   ========================================================================== */
(function () {
    "use strict";

    var IID = window.STACKIT.iid;
    var base = "/game/" + IID;

    var boardEl = document.getElementById("board");
    var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    var RING_STEP = 95;   // ms between cascade rings
    var busy = false;     // true while a move / AI turn is in flight

    /* ------------------------------------------------------------- helpers -- */

    function grid()  { return document.getElementById("board-grid"); }
    function cells() { return grid() ? grid().querySelectorAll(".cell") : []; }

    // map id -> {value, owner} of the currently rendered board
    function snapshot() {
        var snap = {};
        cells().forEach(function (c) {
            snap[c.id] = {
                value: parseInt(c.dataset.value, 10) || 0,
                owner: c.dataset.owner
            };
        });
        return snap;
    }

    function applyState(cell, value, owner) {
        cell.dataset.value = value;
        cell.dataset.owner = owner;
        cell.style.setProperty("--v", value);
        var pip = cell.querySelector(".pip");
        if (pip) pip.textContent = value ? value : "";
    }

    function post(path, data) {
        var body = new URLSearchParams(data || {});
        return fetch(base + path, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: body.toString()
        }).then(function (r) { return r.text(); });
    }

    /* --------------------------------------------------------- board swap -- */

    // Replace the board partial, then optionally play the topple animation
    // that morphs `before` (the previous snapshot) into the freshly rendered
    // final state, rippling outward from the move origin.
    function swap(html, before) {
        boardEl.innerHTML = html;
        bindTiles();
        updateScore();
        maybeWin();
        reflectBusy();
        if (before) animateTopple(before);
    }

    function animateTopple(before) {
        var g = grid();
        if (!g) return;
        var origin = g.dataset.lm;                 // "x-y" of the last move
        if (reduceMotion || !origin) return;

        var all = {};
        cells().forEach(function (c) { all[c.id] = c; });

        var oc = origin.split("-");
        var ox = parseInt(oc[0], 10), oy = parseInt(oc[1], 10);

        // Which tiles changed? Freeze the changed ones (except origin) back to
        // their OLD look; they will flip to the new look on their ring's tick.
        var waves = {};   // distance -> [cells]
        Object.keys(all).forEach(function (id) {
            var cell = all[id];
            var prev = before[id];
            if (!prev) return;
            var changed = (parseInt(cell.dataset.value, 10) !== prev.value) ||
                          (cell.dataset.owner !== prev.owner);
            if (!changed || id === origin) return;

            var p = id.split("-");
            var d = Math.abs(parseInt(p[0], 10) - ox) + Math.abs(parseInt(p[1], 10) - oy);

            // stash final look, render the old look for now
            cell._final = { value: parseInt(cell.dataset.value, 10), owner: cell.dataset.owner };
            applyState(cell, prev.value, prev.owner);
            (waves[d] = waves[d] || []).push(cell);
        });

        // origin pops first
        var originCell = all[origin];
        if (originCell) {
            originCell.classList.add("pop");
            originCell.addEventListener("animationend", function h() {
                originCell.classList.remove("pop");
                originCell.removeEventListener("animationend", h);
            });
        }

        // outward ripple: each ring flips to its final look, staggered
        Object.keys(waves).map(Number).sort(function (a, b) { return a - b; })
            .forEach(function (d) {
                setTimeout(function () {
                    waves[d].forEach(function (cell) {
                        if (cell._final) { applyState(cell, cell._final.value, cell._final.owner); cell._final = null; }
                        cell.classList.remove("flip");
                        void cell.offsetWidth;          // restart animation
                        cell.classList.add("flip");
                        cell.addEventListener("animationend", function h() {
                            cell.classList.remove("flip");
                            cell.removeEventListener("animationend", h);
                        });
                    });
                }, d * RING_STEP);
            });

        g.classList.remove("settling"); void g.offsetWidth; g.classList.add("settling");
    }

    /* ---------------------------------------------------------- scoreboard - */

    function updateScore() {
        var counts = { p1: 0, p2: 0 };
        cells().forEach(function (c) {
            if (c.dataset.owner === "p1") counts.p1++;
            else if (c.dataset.owner === "p2") counts.p2++;
        });
        var t1 = document.getElementById("tally-p1");
        var t2 = document.getElementById("tally-p2");
        if (t1) t1.textContent = counts.p1;
        if (t2) t2.textContent = counts.p2;
    }

    function maybeWin() {
        var g = grid();
        var banner = document.getElementById("banner");
        if (!g || !banner) return;
        var w = g.dataset.win;   // "0" | "1" | "2"
        if (w === "1" || w === "2") {
            var who = w === "1" ? "p1" : "p2";
            banner.className = "banner show " + who;
            banner.querySelector(".msg").textContent =
                (w === "1" ? "You win!" : "The AI wins!") + " Every cell captured.";
        } else {
            banner.className = "banner";
        }
    }

    /* ----------------------------------------------------- turn / busy UI -- */

    function humansTurn() {
        var g = grid();
        return g && g.dataset.current === "p1";
    }

    function setBusy(on) {
        busy = on;
        reflectBusy();
    }

    // Apply the current `busy` state to whatever board DOM is present now
    // (re-applied after every swap, since the board partial is replaced).
    function reflectBusy() {
        var g = grid();
        if (g) g.classList.toggle("busy", busy);
        ["btn-new", "btn-set", "btn-undo"].forEach(function (id) {
            var b = document.getElementById(id);
            if (b) b.disabled = busy;
        });
        var status = document.getElementById("status");
        if (status) status.classList.toggle("busy", busy);
        var p2 = document.getElementById("pcard-p2");
        if (p2) p2.classList.toggle("thinking", busy);
    }

    /* --------------------------------------------------------- interaction - */

    function bindTiles() {
        var human = humansTurn();
        cells().forEach(function (c) {
            var mine = c.dataset.owner === "p1" || c.dataset.owner === "empty";
            if (human && mine) {
                c.classList.add("playable");
                c.classList.remove("locked");
                c.addEventListener("click", onTileClick);
            } else {
                c.classList.add("locked");
            }
        });
    }

    function onTileClick(e) {
        if (busy || !humansTurn()) return;
        var before = snapshot();
        setBusy(true);
        post("/move", { move: this.id })
            .then(function (html) {
                swap(html, before);
                return queryAI();
            })
            .catch(function () { setBusy(false); });
    }

    function queryAI() {
        if (grid().dataset.win !== "0") { setBusy(false); return Promise.resolve(); }
        var before = snapshot();
        var status = document.getElementById("status-text");
        var secs = parseInt(grid().dataset.thinking, 10) || 0;
        var left = secs, timer = null;
        if (status) {
            status.textContent = "AI is thinking… " + left + "s";
            timer = setInterval(function () {
                left = Math.max(0, left - 1);
                status.textContent = "AI is thinking… " + left + "s";
            }, 1000);
        }
        return fetch(base + "/move", { method: "GET" })
            .then(function (r) { return r.text(); })
            .then(function (html) {
                if (timer) clearInterval(timer);
                if (status) status.textContent = "";
                swap(html, before);
                setBusy(false);
            })
            .catch(function () {
                if (timer) clearInterval(timer);
                if (status) status.textContent = "";
                setBusy(false);
            });
    }

    /* ------------------------------------------------------- menu buttons -- */

    document.getElementById("btn-new").addEventListener("click", function () {
        var size = document.getElementById("size").value;
        setBusy(true);
        post("/new", { x: size, y: size }).then(function (html) {
            swap(html, null); setBusy(false);
        }).catch(function () { setBusy(false); });
    });

    document.getElementById("btn-set").addEventListener("click", function () {
        setBusy(true);
        post("/set", {
            thinking_time: document.getElementById("thinking_time").value,
            max_depth: document.getElementById("max_depth").value,
            ai: document.getElementById("ais").value
        }).then(function (html) { swap(html, null); setBusy(false); })
          .catch(function () { setBusy(false); });
    });

    document.getElementById("btn-undo").addEventListener("click", function () {
        setBusy(true);
        post("/undo", {}).then(function (html) { swap(html, null); setBusy(false); })
                         .catch(function () { setBusy(false); });
    });

    /* ------------------------------------------------------------- init ---- */
    bindTiles();
    updateScore();
    maybeWin();
})();
