/* ==========================================================================
   StackIt — front end.

   Talks to the JSON API in server.py. Renders the board from state, plays the
   real cascade frames the server records (no guessing), and draws the engine's
   analysis: principal variation, candidate moves, and for AlphaZero the raw
   network policy as a heat overlay.
   ========================================================================== */
(function () {
"use strict";

var FRAME_MS = 110;          // pause between cascade frames
var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

var POLL_MS = 1000;          // how often an online browser asks for news
var POLL_HIDDEN_MS = 4000;   // ... and how often when the tab is in the background

var S = {                    // client state
    iid: null,
    state: null,
    analysis: null,
    net: null,
    setup: { mode: "hva", ai: "alphabeta", seat: 1 },
    engines: [],
    limits: { max_board: 8, max_board_humans: 12, max_players: 5 },
    busy: false,
    heat: true,
    token: null,             // proves which seat is ours in an online game
    drawn: 0,                // moves already painted, so polling knows what it missed
    pollTimer: null
};

var $ = function (id) { return document.getElementById(id); };
function plural(n, word) { return n + " " + word + (n === 1 ? "" : "s"); }

/* ------------------------------------------------------------------ api -- */

function api(path, opts) {
    opts = opts || {};
    var init = { method: opts.method || "GET", headers: {} };
    if (opts.body) {
        init.headers["Content-Type"] = "application/json";
        init.body = JSON.stringify(opts.body);
    }
    return fetch(path, init).then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (j) {
            if (!r.ok) throw new Error(j.error || ("Request failed (" + r.status + ")"));
            return j;
        });
    });
}

var toastTimer = null;
function toast(msg) {
    var t = $("toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.classList.remove("show"); }, 3800);
}

/* ---------------------------------------------------------------- setup -- */

function paintSetup() {
    document.querySelectorAll("#mode-choice .choice").forEach(function (b) {
        b.classList.toggle("on", b.dataset.mode === S.setup.mode);
    });
    document.querySelectorAll("#seat-field .choice").forEach(function (b) {
        b.classList.toggle("on", parseInt(b.dataset.seat, 10) === S.setup.seat);
    });
    document.querySelectorAll("#engine-choice .choice").forEach(function (b) {
        b.classList.toggle("on", b.dataset.engine === S.setup.ai);
    });
    var mode = S.setup.mode, vsAI = mode === "hva";
    $("opponent-field").hidden = !vsAI;
    $("seat-field").hidden = !vsAI;
    $("time-field").hidden = !vsAI;
    $("players-field").hidden = vsAI;
    $("name-field").hidden = mode !== "online";
    $("btn-start").textContent = mode === "online" ? "Create the game" : "Start game";
    applySizeCap();
}

/** Only an engine has to search the board, so a humans-only game is allowed a
 *  bigger one. Keep the form honest about which cap is in force. */
function applySizeCap() {
    var cap = S.setup.mode === "hva" ? S.limits.max_board : S.limits.max_board_humans;
    var size = $("in-size");
    size.max = cap;
    if (+size.value > cap) size.value = cap;
    $("size-hint").textContent = size.value + " by " + size.value + " grid, up to "
                               + cap + " by " + cap + ".";
}

function renderEngines(list) {
    S.engines = list;
    var wrap = $("engine-choice");
    wrap.innerHTML = "";
    list.forEach(function (e) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "choice";
        b.dataset.engine = e.id;
        b.disabled = !e.available;
        var sub = e.available ? e.blurb : e.reason;
        b.innerHTML = '<span class="choice-title"></span><span class="choice-sub"></span>';
        b.querySelector(".choice-title").textContent = e.name;
        b.querySelector(".choice-sub").textContent = sub;
        b.addEventListener("click", function () {
            S.setup.ai = e.id;
            paintSetup();
        });
        wrap.appendChild(b);
    });
    // If the chosen engine is unavailable at this size, fall back to the first
    // one that is, so "Start" never fails for a reason the user cannot see.
    var chosen = list.filter(function (e) { return e.id === S.setup.ai; })[0];
    if (!chosen || !chosen.available) {
        var ok = list.filter(function (e) { return e.available; })[0];
        if (ok) S.setup.ai = ok.id;
    }
    paintSetup();
}

function refreshEngines() {
    var size = parseInt($("in-size").value, 10) || 5;
    applySizeCap();
    return api("/api/engines?size=" + size)
        .then(function (j) {
            applyLimits(j.limits);
            renderEngines(j.engines);
        })
        .catch(function (e) { toast(e.message); });
}

/** The server caps board size and thinking time; mirror those caps in the form
 *  so nobody picks a number that is silently clamped on submit. */
function applyLimits(lim) {
    if (!lim) return;
    S.limits = lim;
    var time = $("in-time"), players = $("in-players");
    time.max = lim.max_thinking_time;
    players.max = lim.max_players;
    if (+players.value > lim.max_players) players.value = lim.max_players;
    if (+time.value > lim.max_thinking_time) time.value = lim.max_thinking_time;
    $("time-hint").textContent =
        "Seconds per computer move, up to " + lim.max_thinking_time +
        ". More time, stronger play.";
    applySizeCap();
}

/* ---------------------------------------------------------------- board -- */

/** Cells are a fixed 62px up to a 7-wide board; wider than that they shrink
 *  so a 12x12 grid still fits a laptop screen and a phone. */
function fitBoard() {
    var st = S.state;
    if (!st) return;
    var g = $("board-grid");
    var gap = 9, room = Math.min(window.innerWidth - 44, 760);
    var size = Math.floor((room - (st.size_x - 1) * gap) / st.size_x);
    g.style.setProperty("--cell", Math.max(26, Math.min(62, size)) + "px");
}

function buildBoard(st) {
    var g = $("board-grid");
    g.style.setProperty("--cols", st.size_x);
    g.innerHTML = "";
    for (var y = 0; y < st.size_y; y++) {
        for (var x = 0; x < st.size_x; x++) {
            var c = document.createElement("div");
            c.className = "cell";
            c.id = "c-" + x + "-" + y;
            c.dataset.x = x;
            c.dataset.y = y;
            c.innerHTML = '<span class="heat"></span><span class="pip"></span>';
            c.addEventListener("click", onCellClick);
            g.appendChild(c);
        }
    }
}

function cellAt(x, y) { return $("c-" + x + "-" + y); }

function paintCells(cells, sizeX) {
    cells.forEach(function (pair, i) {
        var c = cellAt(i % sizeX, Math.floor(i / sizeX));
        if (!c) return;
        c.dataset.owner = pair[1];
        c.dataset.value = pair[0];
        c.querySelector(".pip").textContent = pair[0] ? pair[0] : "";
    });
}

/** Play the server's recorded cascade, then settle on the final position. */
function animate(frames, origin, sizeX) {
    if (reduceMotion || !frames || frames.length < 2) return Promise.resolve();
    var oc = origin ? cellAt(origin[0], origin[1]) : null;
    if (oc) {
        oc.classList.remove("pop");
        void oc.offsetWidth;
        oc.classList.add("pop");
    }
    // Every frame gets its own hold, the last one included — otherwise the
    // final ring is overwritten by the settled position before it is visible.
    return frames.reduce(function (chain, frame, i) {
        return chain.then(function () {
            return new Promise(function (res) {
                setTimeout(function () {
                    var before = {};
                    document.querySelectorAll(".cell").forEach(function (c) {
                        before[c.id] = c.dataset.value + "/" + c.dataset.owner;
                    });
                    paintCells(frame, sizeX);
                    document.querySelectorAll(".cell").forEach(function (c) {
                        if (before[c.id] !== c.dataset.value + "/" + c.dataset.owner) {
                            c.classList.remove("flip");
                            void c.offsetWidth;
                            c.classList.add("flip");
                        }
                    });
                    res();
                }, FRAME_MS);
            });
        });
    }, Promise.resolve());
}

/** Whether the person at this browser may move right now. On one screen that
 *  is anyone whose turn it is; online it is only our own seat. */
function myTurn(st) {
    if (!st || st.over) return false;
    if (st.mode === "hva") return st.current_player !== st.ai_seat;
    if (st.mode === "online") return st.started && st.your_seat === st.current_player;
    return true;
}

function onCellClick() {
    if (S.busy || !S.state) return;
    var st = S.state;
    if (st.over) return;
    if (!myTurn(st)) {
        if (st.mode === "online") toast("Not your turn yet.");
        return;
    }
    var x = parseInt(this.dataset.x, 10), y = parseInt(this.dataset.y, 10);
    var legal = st.legal.some(function (m) { return m[0] === x && m[1] === y; });
    if (!legal) { toast("That cell belongs to somebody else."); return; }

    setBusy(true);
    api("/api/game/" + S.iid + "/move",
        { method: "POST", body: { x: x, y: y, token: S.token } })
        .then(function (j) {
            return animate(j.frames, [x, y], j.state.size_x).then(function () {
                S.drawn = j.state.move_count;
                applyState(j.state);
                return maybeAIMove();
            });
        })
        .catch(function (e) { toast(e.message); })
        .then(function () { setBusy(false); });
}

function maybeAIMove() {
    if (!S.state || !S.state.is_ai_turn) return Promise.resolve();
    setThinking(true);
    return api("/api/game/" + S.iid + "/ai", { method: "POST" })
        .then(function (j) {
            setThinking(false);
            S.analysis = j.analysis;
            renderAnalysis(j.analysis);
            return animate(j.frames, j.move, j.state.size_x).then(function () {
                applyState(j.state);
                // The engine may move again if the human has no legal reply.
                return maybeAIMove();
            });
        })
        .catch(function (e) {
            setThinking(false);
            toast(e.message);
        });
}

/* --------------------------------------------------------------- render -- */

/** One card per seat. Rebuilt only when the line-up changes, so a poll that
 *  brings nothing new does not make the scoreboard flicker. */
function renderScoreboard(st) {
    var wrap = $("scoreboard");
    var sig = st.players.map(function (p) { return p.seat + ":" + p.name; }).join("|");
    if (wrap.dataset.sig !== sig) {
        wrap.dataset.sig = sig;
        wrap.classList.toggle("many", st.players.length > 2);
        wrap.innerHTML = "";
        st.players.forEach(function (p) {
            var card = document.createElement("div");
            card.className = "pcard";
            card.id = "pcard-" + p.seat;
            card.dataset.p = p.seat;
            card.innerHTML = '<span class="dot"></span><span class="meta">' +
                '<span class="name"></span><span class="sub"></span></span>' +
                '<span class="thinking-badge">thinking' +
                '<span class="dots"><i></i><i></i><i></i></span></span>';
            wrap.appendChild(card);
        });
    }
    st.players.forEach(function (p) {
        var card = $("pcard-" + p.seat);
        card.querySelector(".name").textContent = p.name;
        card.querySelector(".sub").textContent = !p.joined
            ? "not here yet"
            : p.out ? "knocked out"
            : plural(p.cells, "cell") + " · " + plural(p.blocks, "block");
        card.classList.toggle("active", !st.over && st.current_player === p.seat);
        card.classList.toggle("out", !!p.out);
        card.classList.toggle("empty", !p.joined);
        card.classList.toggle("you", !!p.is_you);
    });
}

function whoSpan(st, seat) {
    var p = st.players[seat - 1];
    return '<span class="who-' + seat + '">' +
           (p ? p.name : "Player " + seat) + "</span>";
}

function applyState(st) {
    // Also rebuild when the grid is empty: an online game keeps polling state
    // while it sits in the lobby, so by the time the game screen appears the
    // sizes already match and only the missing cells give it away.
    var grid = $("board-grid");
    var rebuild = !S.state || S.state.size_x !== st.size_x
               || S.state.size_y !== st.size_y
               || grid.childElementCount !== st.size_x * st.size_y;
    S.state = st;
    if (rebuild) { buildBoard(st); fitBoard(); }
    paintCells(st.cells, st.size_x);

    // last-move marker
    document.querySelectorAll(".cell.lastmove").forEach(function (c) {
        c.classList.remove("lastmove");
    });
    if (st.last_move) {
        var lm = cellAt(st.last_move[0], st.last_move[1]);
        if (lm) lm.classList.add("lastmove");
    }

    // which cells the person at this browser may click right now
    var humansTurn = myTurn(st);
    var legal = {};
    st.legal.forEach(function (m) { legal[m[0] + "-" + m[1]] = true; });
    document.querySelectorAll(".cell").forEach(function (c) {
        var ok = humansTurn && legal[c.dataset.x + "-" + c.dataset.y];
        c.classList.toggle("playable", !!ok);
    });

    renderScoreboard(st);

    // turn line
    var turn = $("turn-text");
    if (st.over) {
        turn.textContent = "Game over.";
    } else if (st.mode === "hva" && st.current_player === st.ai_seat) {
        turn.textContent = st.ai_name + " is choosing a move…";
    } else if (myTurn(st)) {
        turn.textContent = "Your turn. Click a grey cell or one of your own.";
    } else {
        turn.innerHTML = whoSpan(st, st.current_player) + " to move.";
    }
    // S.busy still wins: a repaint mid-turn must not re-enable the buttons.
    $("btn-undo").disabled = S.busy || !st.can_undo || st.over;

    // win overlay
    var ov = $("win-overlay");
    if (st.over) {
        var msg, winner = st.players[st.winner - 1];
        if (!st.winner) {
            msg = "It is a draw.";
        } else if (st.mode === "hva") {
            msg = st.winner === st.human_seat ? "You win!" : st.ai_name + " wins.";
        } else if (st.mode === "online" && st.winner === st.your_seat) {
            msg = "You win!";
        } else {
            msg = (winner ? winner.name : "Player " + st.winner) + " wins!";
        }
        $("win-msg").textContent = msg;
        ov.classList.add("show");
    } else {
        ov.classList.remove("show");
    }

    $("analysis").hidden = st.mode !== "hva";
    refreshNetRead();
}

function setBusy(on) {
    S.busy = on;
    var g = $("board-grid");
    if (g) g.classList.toggle("busy", on);
    ["btn-undo", "btn-new"].forEach(function (id) { $(id).disabled = on; });
    if (!on && S.state) $("btn-undo").disabled = !S.state.can_undo || S.state.over;
}

function setThinking(on) {
    if (!S.state || S.state.mode !== "hva") return;
    var card = $("pcard-" + S.state.ai_seat);
    if (card) card.classList.toggle("thinking", on);
    if (on) {
        $("eval-cap").textContent = "thinking…";
        $("an-when").textContent = "";
    }
}

/* ------------------------------------------------------------- analysis -- */

function mvText(m) { return "col " + m[0] + ", row " + m[1]; }
function mvShort(m) { return m[0] + "," + m[1]; }

function peek(m, on) {
    var c = cellAt(m[0], m[1]);
    if (c) c.classList.toggle("peek", on);
}

function renderAnalysis(a) {
    S.analysis = a;
    if (!a) return;

    $("an-engine").textContent = a.name;
    $("an-when").textContent = a.elapsed + "s of thinking";

    // headline evaluation
    $("eval-num").textContent = a.headline.text;
    $("eval-cap").textContent = a.headline.caption;
    var fill = $("eval-fill"), pct;
    if (a.headline.kind === "score") {
        // material score: map roughly -20..+20 onto the bar
        var v = Math.max(-20, Math.min(20, a.headline.value || 0));
        pct = (v + 20) / 40 * 100;
    } else {
        pct = (a.headline.value || 0) * 100;
    }
    fill.style.width = pct + "%";
    fill.style.background = S.state && S.state.ai_seat === 2 ? "var(--p2)" : "var(--p1)";

    // stat chips
    var chips = [];
    if (a.depth !== null && a.depth !== undefined) {
        chips.push(["looked ahead", a.depth + " moves"]);
    }
    if (a.engine === "alphazero") {
        chips.push(["positions imagined", a.nodes]);
    } else if (a.nodes) {
        chips.push(["positions checked", a.nodes.toLocaleString()]);
    }
    $("an-stats").innerHTML = chips.map(function (c) {
        return '<span class="stat">' + c[0] + ' <b>' + c[1] + '</b></span>';
    }).join("");

    renderPV(a);
    renderCandidates(a);
    renderLadder(a);
}

function renderPV(a) {
    var wrap = $("pv-list");
    $("pv-block").hidden = false;
    wrap.innerHTML = "";
    if (!a.pv || !a.pv.length) {
        wrap.innerHTML = '<span class="pv-empty">It did not look far enough ahead to say.</span>';
        return;
    }
    var seat = S.state ? S.state.ai_seat : 2;
    a.pv.forEach(function (m, i) {
        var who = ((i % 2) === 0) ? seat : (seat === 1 ? 2 : 1);
        var el = document.createElement("span");
        el.className = "pv-step";
        el.dataset.seat = who;
        el.innerHTML = '<span class="n"></span><span class="t"></span>';
        el.querySelector(".n").textContent = i + 1;
        el.querySelector(".t").textContent = mvShort(m);
        el.title = (i === 0 ? "it plays " : (who === seat ? "it plays " : "it expects you to play "))
                 + mvText(m);
        el.addEventListener("mouseenter", function () { peek(m, true); });
        el.addEventListener("mouseleave", function () { peek(m, false); });
        wrap.appendChild(el);
    });
}

function renderCandidates(a) {
    var block = $("cand-block"), wrap = $("cand-list");
    if (!a.candidates || !a.candidates.length) { block.hidden = true; return; }
    block.hidden = false;
    $("cand-hint").textContent = a.engine === "alphazero"
        ? "How often the search chose each move, and how good it looked."
        : "The moves it tried, best first.";
    var maxW = a.candidates.reduce(function (m, c) { return Math.max(m, c.weight || 0); }, 0) || 1;
    wrap.innerHTML = "";
    a.candidates.forEach(function (c, i) {
        var row = document.createElement("div");
        row.className = "cand" + (i === 0 ? " best" : "");
        row.innerHTML = '<span class="fill"></span><span class="mv"></span>' +
                        '<span class="val"></span><span class="wt"></span>';
        row.querySelector(".fill").style.width = ((c.weight || 0) / maxW * 100) + "%";
        row.querySelector(".mv").textContent = mvShort(c.move);
        row.querySelector(".val").textContent = c.value_text;
        row.querySelector(".wt").textContent = c.weight
            ? (a.engine === "alphazero" ? c.weight + " visits" : c.weight + " games")
            : "";
        row.title = mvText(c.move);
        row.addEventListener("mouseenter", function () { peek(c.move, true); });
        row.addEventListener("mouseleave", function () { peek(c.move, false); });
        wrap.appendChild(row);
    });
}

/** Ask the net what it makes of the position currently on screen. This is a
 *  single forward pass with no search, and it always describes the board the
 *  user is looking at — unlike the search analysis, which is one move stale. */
function refreshNetRead() {
    var block = $("net-block");
    if (!S.state || S.state.ai !== "alphazero" || S.state.mode !== "hva" || S.state.over) {
        S.net = null; block.hidden = true; applyHeat(); return Promise.resolve();
    }
    return api("/api/game/" + S.iid + "/netread")
        .then(function (j) {
            S.net = j.net;
            if (!j.net) { block.hidden = true; applyHeat(); return; }
            block.hidden = false;
            var st = S.state;
            var mine = j.net.for_player === st.human_seat;
            $("net-hint").textContent = mine
                ? "This is where the net would play if it sat in your seat. Brighter means keener."
                : "This is where the net wants to play next. Brighter means keener.";
            $("net-label").textContent = mine
                ? "Your chance to win, in its eyes"
                : "Its chance to win";
            $("net-wp").textContent = j.net.win_prob_text;
            applyHeat();
        })
        .catch(function () { S.net = null; block.hidden = true; applyHeat(); });
}

function applyHeat() {
    var net = S.net;
    document.querySelectorAll(".cell").forEach(function (c) {
        c.style.removeProperty("--heat");
        var n = c.querySelector(".heat-num");
        if (n) n.remove();
    });
    if (!S.heat || !net || !net.policy || !net.policy.length) return;
    var max = net.policy.reduce(function (m, p) { return Math.max(m, p[2]); }, 0) || 1;
    net.policy.forEach(function (p) {
        var c = cellAt(p[0], p[1]);
        if (!c) return;
        var rel = p[2] / max;
        if (rel < 0.06) return;
        c.style.setProperty("--heat", (0.15 + rel * 0.85).toFixed(2));
        if (rel > 0.18) {
            var n = document.createElement("span");
            n.className = "heat-num";
            n.textContent = Math.round(p[2] * 100) + "%";
            c.appendChild(n);
        }
    });
}

function renderLadder(a) {
    var block = $("ladder-block"), body = $("ladder-body");
    if (!a.ladder || !a.ladder.length) { block.hidden = true; return; }
    block.hidden = false;
    body.innerHTML = "";
    a.ladder.slice().reverse().forEach(function (r) {
        var tr = document.createElement("tr");
        tr.innerHTML = '<td class="d"></td><td class="s"></td><td class="l"></td>';
        tr.children[0].textContent = r.depth + " deep";
        tr.children[1].textContent = (r.score > 0 ? "+" : "") + r.score;
        tr.children[2].textContent = (r.pv || []).map(mvShort).join(" → ");
        body.appendChild(tr);
    });
}

/* ------------------------------------------------------------- controls -- */

function showScreen(name) {
    $("setup").hidden = name !== "setup";
    $("lobby").hidden = name !== "lobby";
    $("game").hidden = name !== "game";
}

function enterGameScreen(st) {
    showScreen("game");
    S.analysis = null;
    ["pv-block", "cand-block", "net-block", "ladder-block"].forEach(function (id) {
        $(id).hidden = true;
    });
    $("eval-num").textContent = "—";
    $("eval-cap").textContent = "no moves yet";
    $("eval-fill").style.width = "0";
    $("an-stats").innerHTML = "";
    $("an-engine").textContent = st.ai_name;
    $("an-when").textContent = "";
    applyState(st);
}

function startGame() {
    var mode = S.setup.mode;
    var body = {
        size: parseInt($("in-size").value, 10) || 5,
        mode: mode,
        ai: S.setup.ai,
        human_seat: S.setup.seat,
        players: parseInt($("in-players").value, 10) || 2,
        thinking_time: parseInt($("in-time").value, 10) || 3,
        name: $("in-name").value
    };
    $("setup-error").textContent = "";
    $("btn-start").disabled = true;
    api("/api/game", { method: "POST", body: body })
        .then(function (j) {
            S.iid = j.state.iid;
            S.token = j.token || null;
            S.drawn = 0;
            if (mode === "online") {
                S.state = j.state;
                rememberSeat();
                renderLobby(j.state);
                showScreen("lobby");
                startPolling();
                return;
            }
            enterGameScreen(j.state);
            setBusy(true);
            return maybeAIMove().then(function () { setBusy(false); });
        })
        .catch(function (e) { $("setup-error").textContent = e.message; })
        .then(function () { $("btn-start").disabled = false; });
}

function showSetup() {
    stopPolling();
    forgetSeat();
    S.iid = null;
    S.token = null;
    S.state = null;
    S.drawn = 0;
    showScreen("setup");
    refreshEngines();
}

/* --------------------------------------------------------------- online -- */

/* A seat belongs to whoever holds its token, and the token only lives in this
   browser. Park it so a refresh, a stray back button or a phone locking itself
   does not lose the seat. Private browsing may refuse the write; that is fine,
   it only costs the resume. */
var SEAT_KEY = "stackit.seat";

function rememberSeat() {
    try {
        localStorage.setItem(SEAT_KEY, JSON.stringify({ iid: S.iid, token: S.token }));
    } catch (e) { /* storage disabled — nothing to resume from, no harm done */ }
}

function forgetSeat() {
    try { localStorage.removeItem(SEAT_KEY); } catch (e) { /* as above */ }
}

/** Walk back into the game this browser was in, if it is still going. */
function resumeSeat() {
    var saved = null;
    try { saved = JSON.parse(localStorage.getItem(SEAT_KEY)); } catch (e) { return; }
    if (!saved || !saved.iid || !saved.token) return;
    api("/api/game/" + saved.iid + "?token=" + encodeURIComponent(saved.token))
        .then(function (j) {
            // The seat is gone if the game expired, ended, or never knew us.
            if (j.state.your_seat === null || j.state.over) { forgetSeat(); return; }
            S.iid = saved.iid;
            S.token = saved.token;
            S.drawn = j.state.move_count;
            S.state = j.state;
            if (j.state.started) enterGameScreen(j.state);
            else { renderLobby(j.state); showScreen("lobby"); }
            startPolling();
        })
        .catch(forgetSeat);
}

function shareLink(code) {
    return location.origin + location.pathname + "?join=" + code;
}

function renderLobby(st) {
    $("lobby-code").textContent = st.code || "—";
    var wrap = $("lobby-seats");
    wrap.innerHTML = "";
    st.players.forEach(function (p) {
        var row = document.createElement("div");
        row.className = "lobby-seat" + (p.joined ? "" : " empty");
        row.innerHTML = '<span class="swatch p' + p.seat + '"></span><span class="who"></span>';
        row.querySelector(".who").textContent = p.joined ? p.name : "waiting…";
        wrap.appendChild(row);
    });
    var missing = st.num_players - st.seats_taken;
    $("lobby-waiting").innerHTML = missing > 0
        ? "Waiting for " + plural(missing, "more player") +
          '<span class="dots"><i></i><i></i><i></i></span>'
        : "Everyone is here.";
    // Only the host may cut the wait short, and only with somebody to play.
    $("btn-start-now").hidden = !(st.is_host && st.seats_taken >= 2 && missing > 0);
}

function joinGame() {
    var code = ($("in-code").value || "").trim().toUpperCase();
    if (code.length !== 4) {
        $("join-error").textContent = "The code is four letters.";
        return;
    }
    $("join-error").textContent = "";
    $("btn-join").disabled = true;
    api("/api/join", { method: "POST",
                       body: { code: code, name: $("in-join-name").value } })
        .then(function (j) {
            S.iid = j.state.iid;
            S.token = j.token;
            S.drawn = 0;
            S.state = j.state;
            rememberSeat();
            // Drop the ?join= from the address bar. It has done its job, and
            // leaving it there would make a refresh try to join again — with a
            // seat we already hold, in a game that has already started.
            try { history.replaceState(null, "", location.pathname); } catch (e) {}
            if (j.state.started) {
                enterGameScreen(j.state);
            } else {
                renderLobby(j.state);
                showScreen("lobby");
            }
            startPolling();
        })
        .catch(function (e) { $("join-error").textContent = e.message; })
        .then(function () { $("btn-join").disabled = false; });
}

/* Polling, not sockets: a friendly game makes a move every few seconds, so one
   small request per second per browser is plenty and needs no extra plumbing
   on the server. */
function startPolling() {
    stopPolling();
    S.pollTimer = setTimeout(pollTick, document.hidden ? POLL_HIDDEN_MS : POLL_MS);
}

function stopPolling() {
    if (S.pollTimer) { clearTimeout(S.pollTimer); S.pollTimer = null; }
}

function pollTick() {
    S.pollTimer = null;
    if (!S.iid || !S.token) return;
    if (S.busy) { startPolling(); return; }   // mid-animation; ask again shortly
    api("/api/game/" + S.iid + "?since=" + S.drawn +
        "&token=" + encodeURIComponent(S.token))
        .then(applyPoll)
        .catch(function (e) { toast(e.message); })
        .then(function () {
            if (S.state && S.state.over) return;   // nothing else will happen
            startPolling();
        });
}

function applyPoll(j) {
    var st = j.state;
    if (!st.started) {
        S.state = st;
        renderLobby(st);
        showScreen("lobby");
        return;
    }
    if (!$("lobby").hidden) enterGameScreen(st);
    if (st.move_count === S.drawn) { applyState(st); return; }

    var events = j.events || [];
    if (!events.length) {          // too far behind to replay: just take the board
        S.drawn = st.move_count;
        applyState(st);
        return;
    }
    setBusy(true);
    return events.reduce(function (chain, e) {
        return chain.then(function () {
            return animate(e.frames, e.move, st.size_x).then(function () {
                S.drawn = e.n;
            });
        });
    }, Promise.resolve()).then(function () {
        S.drawn = st.move_count;
        applyState(st);
        setBusy(false);
    });
}

function bind() {
    document.querySelectorAll("#mode-choice .choice").forEach(function (b) {
        b.addEventListener("click", function () { S.setup.mode = b.dataset.mode; paintSetup(); });
    });
    document.querySelectorAll("#seat-field .choice").forEach(function (b) {
        b.addEventListener("click", function () {
            S.setup.seat = parseInt(b.dataset.seat, 10);
            paintSetup();
        });
    });
    document.querySelectorAll(".step").forEach(function (b) {
        b.addEventListener("click", function () {
            var input = $(b.dataset.target);
            var v = (parseInt(input.value, 10) || 0) + parseInt(b.dataset.step, 10);
            var lo = parseInt(input.min, 10), hi = parseInt(input.max, 10);
            input.value = Math.max(lo, Math.min(hi, v));
            input.dispatchEvent(new Event("change"));
        });
    });
    $("in-size").addEventListener("change", refreshEngines);
    $("btn-start").addEventListener("click", startGame);
    $("btn-new").addEventListener("click", showSetup);
    $("btn-again").addEventListener("click", showSetup);

    // -- online ---------------------------------------------------------
    $("btn-join").addEventListener("click", joinGame);
    $("in-code").addEventListener("input", function () {
        this.value = this.value.toUpperCase().replace(/[^A-Z0-9]/g, "");
    });
    $("in-code").addEventListener("keydown", function (e) {
        if (e.key === "Enter") joinGame();
    });
    $("btn-copy").addEventListener("click", function () {
        var link = shareLink(S.state && S.state.code);
        var done = function () { toast("Link copied. Send it to your friends."); };
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(link).then(done, function () { prompt("Copy this link:", link); });
        } else {
            prompt("Copy this link:", link);
        }
    });
    $("btn-start-now").addEventListener("click", function () {
        api("/api/game/" + S.iid + "/start", { method: "POST", body: { token: S.token } })
            .then(function (j) { S.state = j.state; enterGameScreen(j.state); })
            .catch(function (e) { toast(e.message); });
    });
    $("btn-lobby-cancel").addEventListener("click", showSetup);

    // Keep the board on screen when the window changes shape, and ease off the
    // polling while the tab is hidden.
    window.addEventListener("resize", fitBoard);
    document.addEventListener("visibilitychange", function () {
        if (S.pollTimer) startPolling();
    });

    $("btn-undo").addEventListener("click", function () {
        if (S.busy || !S.iid) return;
        setBusy(true);
        api("/api/game/" + S.iid + "/undo", { method: "POST", body: { token: S.token } })
            .then(function (j) {
                S.analysis = null;
                ["pv-block", "cand-block", "net-block", "ladder-block"].forEach(function (id) {
                    $(id).hidden = true;
                });
                $("eval-num").textContent = "—";
                $("eval-cap").textContent = "taken back";
                $("eval-fill").style.width = "0";
                $("an-stats").innerHTML = "";
                applyState(j.state);
            })
            .catch(function (e) { toast(e.message); })
            .then(function () { setBusy(false); });
    });

    $("net-heat").addEventListener("change", function () {
        S.heat = this.checked;
        applyHeat();
    });

    $("btn-rules").addEventListener("click", function () { $("rules").hidden = false; });
    $("btn-rules-close").addEventListener("click", function () { $("rules").hidden = true; });
    $("rules").addEventListener("click", function (e) {
        if (e.target === this) this.hidden = true;
    });
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") $("rules").hidden = true;
    });
}

/** A friend's link looks like /?join=WDPL. Fill the code in and put the cursor
 *  on Join — one click, no lobby to hunt through. */
function readJoinLink() {
    var m = /[?&]join=([A-Za-z0-9]{1,8})/.exec(location.search);
    if (!m) return false;
    $("in-code").value = m[1].toUpperCase();
    $("in-join-name").focus();
    $("join-card").scrollIntoView({ behavior: "smooth", block: "center" });
    return true;
}

bind();
paintSetup();
refreshEngines();
// A link to join beats walking back into an old game: the person clicked it.
if (!readJoinLink()) resumeSeat();

})();
