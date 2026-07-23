"""Live training dashboard for StackIt AlphaZero.

Reads the artifacts the training loop writes (checkpoints/metrics.jsonl,
status.json, last_game.json) and serves a single self-contained page: loss and
win-rate curves (hoverable), live counters, an accepted-candidate history table,
and an animated replay of the latest self-play game. Runs independently of
training — start the trainer in one terminal and this in another.

    python3 -m alphazero.dashboard          # http://localhost:8123
"""
import os
import json
import argparse
import threading

import torch
from flask import Flask, jsonify, request

from board import Board
from .config import Config

app = Flask(__name__)
CKPT = Config().ckpt_dir

# ---- interactive "play the best model" state (single local user) ----
_play_lock = threading.Lock()
PLAY = {"game": None}
BEST_PT = os.path.join(CKPT, "best.pt")


def _board_state(board):
    return {"board": [r[:] for r in board.board],
            "player": [r[:] for r in board.player],
            "size": board.size_x, "current": board.current_player,
            "legal": [[x, y] for (x, y) in board.possible_moves()]}


def _status(board):
    w = board.winning_player()
    if w:
        return {"over": True, "winner": w, "reason": "domination"}
    if not board.possible_moves():
        b1, b2 = board.boxes_for(1), board.boxes_for(2)
        return {"over": True, "reason": "no moves",
                "winner": 1 if b1 > b2 else 2 if b2 > b1 else 0}
    return {"over": False}


@app.route("/api/play/new", methods=["POST"])
def play_new():
    """Start a human-vs-best game. Snapshots best.pt into memory NOW, so ongoing
    training overwriting best.pt on disk can't change the opponent mid-game."""
    from .engine import AlphaZero
    with _play_lock:
        if not os.path.exists(BEST_PT):
            return jsonify({"ok": False, "error": "No checkpoint yet — wait for iteration 1."})
        try:
            arch = torch.load(BEST_PT, map_location="cpu", weights_only=False)
            size = arch["arch"]["board_size"]
            version = arch.get("extra", {}).get("version")
            board = Board(size, size)
            engine = AlphaZero(ckpt=BEST_PT, max_sims_cap=200000)
            engine._ensure_loaded(board)          # load snapshot into memory now
        except Exception as exc:                  # mid-write or bad file
            return jsonify({"ok": False, "error": f"could not load checkpoint: {exc}"})
        PLAY["game"] = {"board": board, "engine": engine, "version": version}
        return jsonify({"ok": True, "version": version, "state": _board_state(board)})


@app.route("/api/play/human", methods=["POST"])
def play_human():
    with _play_lock:
        g = PLAY["game"]
        if not g:
            return jsonify({"ok": False, "error": "no game"})
        board = g["board"]
        x, y = int(request.json["x"]), int(request.json["y"])
        if (x, y) not in board.possible_moves():
            return jsonify({"ok": False, "error": "illegal move"})
        board.move(x, y)
        return jsonify({"ok": True, "state": _board_state(board), "status": _status(board)})


@app.route("/api/play/ai", methods=["POST"])
def play_ai():
    with _play_lock:
        g = PLAY["game"]
        if not g:
            return jsonify({"ok": False, "error": "no game"})
        board, engine = g["board"], g["engine"]
        st = _status(board)
        if st["over"]:
            return jsonify({"ok": True, "state": _board_state(board), "status": st})
        info = engine.analyze(board, thinking_time=5.0)
        board.move(*info["move"])
        return jsonify({"ok": True, "state": _board_state(board),
                        "status": _status(board), "ai_move": info["move"], "stats": info})


def _read_json(name, default):
    path = os.path.join(CKPT, name)
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return default          # mid-write; caller retries next poll


@app.route("/api/metrics")
def api_metrics():
    path = Config().metrics_file
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
    return jsonify(rows)


@app.route("/api/status")
def api_status():
    return jsonify(_read_json("status.json", {"phase": "idle"}))


@app.route("/api/ratings")
def api_ratings():
    return jsonify(_read_json("ratings.json", {"versions": [], "champion": None}))


@app.route("/api/game")
def api_game():
    """Reconstruct per-ply board frames from the stored move list by replaying
    them on a real Board (reusing the game's own rules)."""
    rec = _read_json("last_game.json", None)
    if not rec:
        return jsonify({"frames": [], "size": Config().board_size})
    size = rec["size"]
    board = Board(size, size)
    frames = [{"board": [r[:] for r in board.board],
               "player": [r[:] for r in board.player],
               "move": None, "mover": board.current_player}]
    for (x, y) in rec["moves"]:
        mover = board.current_player
        board.move(x, y)
        frames.append({"board": [r[:] for r in board.board],
                       "player": [r[:] for r in board.player],
                       "move": [x, y], "mover": mover})
    return jsonify({"frames": frames, "size": size,
                    "winner": rec.get("winner"), "iter": rec.get("iter"),
                    "plies": rec.get("plies")})


@app.route("/")
def index():
    return PAGE


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>StackIt AlphaZero — Training</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#eaeef7;--bg-2:#f4f7fd;--card:#fff;--ink:#1d2140;--ink-soft:#5a6182;--line:#dbe1ef;
  --p1:#2d7dff;--p1-edge:#1a5ad6;--p1-soft:#e4eeff;
  --p2:#ff8a1e;--p2-edge:#e06d00;--p2-soft:#fff0df;
  --good:#18a058;--bad:#c0392b;--slot:#dde3f0;--slot-line:#c2cbe0;
  --display:"Fredoka",ui-rounded,"Segoe UI Rounded",system-ui,sans-serif;
  --body:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --shadow:0 6px 22px rgba(29,33,64,.14);
}
*{box-sizing:border-box}
body{margin:0;font-family:var(--body);color:var(--ink);
  background:radial-gradient(1100px 600px at 82% -8%,var(--p2-soft) 0%,transparent 55%),
             radial-gradient(1000px 620px at 8% 110%,var(--p1-soft) 0%,transparent 55%),var(--bg);
  background-attachment:fixed;min-height:100vh}
header{padding:22px 28px 6px}
h1{font-family:var(--display);font-weight:700;margin:0;font-size:26px}
h1 .a{color:var(--p1)} h1 .z{color:var(--p2)}
.sub{color:var(--ink-soft);margin:2px 0 0;font-size:14px}
.wrap{display:grid;grid-template-columns:1.4fr 1fr;gap:18px;padding:14px 28px 40px;max-width:1200px}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);padding:16px 18px}
.card h2{font-family:var(--display);font-weight:600;font-size:15px;margin:0 0 10px;color:var(--ink-soft);
  text-transform:uppercase;letter-spacing:.04em}
.statusbar{display:flex;gap:20px;flex-wrap:wrap;align-items:baseline;margin-bottom:6px}
.stat{display:flex;flex-direction:column}
.stat .v{font-family:var(--display);font-weight:700;font-size:22px;line-height:1}
.stat .k{font-size:11px;color:var(--ink-soft);text-transform:uppercase;letter-spacing:.05em;margin-top:3px}
.pill{font-family:var(--display);font-size:13px;font-weight:600;padding:4px 12px;border-radius:999px;
  background:var(--p1-soft);color:var(--p1-edge)}
.pill.done{background:#e6f7ee;color:var(--good)}
.chart{position:relative}
canvas{width:100%;height:auto;display:block}
.chart .tip{position:absolute;pointer-events:none;background:var(--ink);color:#fff;font-size:12px;
  padding:7px 10px;border-radius:9px;box-shadow:var(--shadow);opacity:0;transition:opacity .08s;
  white-space:nowrap;z-index:6;transform:translate(-50%,-112%)}
.chart .tip .th{font-family:var(--display);font-weight:600;margin-bottom:4px;font-size:12px}
.chart .tip .r{display:flex;align-items:center;gap:6px;line-height:1.5}
.chart .vline{position:absolute;top:0;width:1px;background:rgba(29,33,64,.28);opacity:0;pointer-events:none;z-index:5}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink-soft);margin-top:8px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{width:10px;height:10px;border-radius:3px;display:inline-block}
.board{display:grid;gap:8px;margin:6px auto 4px;width:max-content}
.cell{width:52px;height:52px;border-radius:12px;display:flex;align-items:center;justify-content:center;
  font-family:var(--display);font-weight:700;font-size:20px;background:var(--slot);
  border:1px solid var(--slot-line);color:#9aa3bd;transition:background .18s,color .18s,transform .18s}
.cell.p1{background:linear-gradient(160deg,#62a0ff,var(--p1));color:#fff;border-color:var(--p1-edge)}
.cell.p2{background:linear-gradient(160deg,#ffab55,var(--p2));color:#fff;border-color:var(--p2-edge)}
.cell.last{transform:translateY(-3px) scale(1.05);box-shadow:0 5px 0 rgba(0,0,0,.12)}
.gamebar{display:flex;justify-content:space-between;align-items:center;margin-top:8px;font-size:13px;color:var(--ink-soft)}
.winner{font-family:var(--display);font-weight:700}
.winner.p1{color:var(--p1)} .winner.p2{color:var(--p2)}
button{font-family:var(--display);font-weight:600;font-size:13px;border:none;border-radius:10px;
  padding:6px 12px;background:var(--ink);color:#fff;cursor:pointer;box-shadow:0 3px 0 #0d1030}
button:active{transform:translateY(2px);box-shadow:0 1px 0 #0d1030}
.offline{color:var(--bad)}
.logwrap{max-height:240px;overflow-y:auto;margin-top:10px;border-top:1px solid var(--line)}
table.log{width:100%;border-collapse:collapse;font-size:12px}
table.log th{text-align:left;color:var(--ink-soft);font-weight:600;font-size:10.5px;text-transform:uppercase;
  letter-spacing:.03em;padding:5px 6px;position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line)}
table.log td{padding:4px 6px;border-bottom:1px solid #f0f3fa;font-variant-numeric:tabular-nums}
tr.acc td:first-child{box-shadow:inset 3px 0 0 var(--good)}
tr.rej td:first-child{box-shadow:inset 3px 0 0 var(--slot-line)}
.tag{font-family:var(--display);font-weight:700;font-size:10px;padding:1px 6px;border-radius:6px}
.tag.acc{background:#d8f2e2;color:var(--good)} .tag.rej{background:#eef1f7;color:var(--ink-soft)}
.rowhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.rowhead h2{font-family:var(--display);font-weight:600;font-size:15px;color:var(--ink-soft);
  text-transform:uppercase;letter-spacing:.04em}
#mode-btn,#newgame-btn{background:var(--p1);box-shadow:0 3px 0 var(--p1-edge)}
#mode-btn:active,#newgame-btn:active{box-shadow:0 1px 0 var(--p1-edge)}
.cell.legal{cursor:pointer;box-shadow:inset 0 0 0 2px rgba(45,125,255,.55)}
.cell.legal:hover{transform:translateY(-2px);background:var(--p1-soft);color:var(--p1-edge)}
.cell.aimove{box-shadow:0 0 0 3px var(--p2)}
.pstats{margin-top:6px;font-size:13px;color:var(--ink)}
.pstats .sline{margin:3px 0;color:var(--ink-soft)}
.pstats b{color:var(--ink)}
.pstats .pv{font-family:var(--display);font-weight:600;color:var(--p1-edge)}
.heads{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;margin-top:8px}
.heat{display:grid;gap:3px}
.heat .hc{width:30px;height:30px;border-radius:7px;display:flex;align-items:center;justify-content:center;
  font-size:10px;font-weight:600;color:#334;border:1px solid var(--line);font-variant-numeric:tabular-nums}
.vbar{height:14px;border-radius:7px;background:linear-gradient(90deg,#ff8a1e,#e6e9f2 50%,#2d7dff);position:relative;min-width:160px}
.vbar .mk{position:absolute;top:-3px;width:3px;height:20px;background:var(--ink);border-radius:2px;transform:translateX(-50%)}
.mini h4{margin:0 0 4px;font-size:11px;color:var(--ink-soft);text-transform:uppercase;letter-spacing:.04em;font-weight:600}
</style></head><body>
<header>
  <h1><span class="a">Alpha</span><span class="z">Zero</span> · StackIt training</h1>
  <p class="sub">The network plays itself, learns from Monte-Carlo search, and (hopefully) climbs. Live from <code>checkpoints/</code>.</p>
</header>
<div class="statusbar card" style="margin:0 28px">
  <span id="phase" class="pill">idle</span>
  <div class="stat"><span class="v" id="s-iter">–</span><span class="k">iteration</span></div>
  <div class="stat"><span class="v" id="s-elapsed">–</span><span class="k">elapsed</span></div>
  <div class="stat"><span class="v" id="s-speed">–</span><span class="k">avg / iter</span></div>
  <div class="stat"><span class="v" id="s-buf">–</span><span class="k">buffer</span></div>
  <div class="stat"><span class="v" id="s-elo">–</span><span class="k">best elo</span></div>
  <div class="stat"><span class="v" id="s-rand">–</span><span class="k">win vs random</span></div>
  <div class="stat"><span class="v" id="s-ab">–</span><span class="k">win vs alphabeta</span></div>
  <div class="stat"><span class="v" id="s-extra">–</span><span class="k" id="s-extra-k">phase</span></div>
</div>
<div class="wrap">
  <div style="display:flex;flex-direction:column;gap:18px">
    <div class="card">
      <h2>Elo rating <span style="text-transform:none;font-weight:500;color:#c2cbe0">· AlphaBeta = 0</span></h2>
      <div class="chart" id="w-elo"><canvas id="c-elo" width="640" height="230"></canvas>
        <div class="vline"></div><div class="tip"></div></div>
      <div class="legend">
        <span><i class="dot" style="background:var(--good)"></i>current model</span>
        <span><i class="dot" style="background:var(--p1)"></i>champion (best)</span>
        <span><i class="dot" style="background:var(--p2)"></i>reference bars (dashed)</span>
      </div>
    </div>
    <div class="card">
      <h2>Win rate</h2>
      <div class="chart" id="w-win"><canvas id="c-win" width="640" height="260"></canvas>
        <div class="vline"></div><div class="tip"></div></div>
      <div class="legend">
        <span><i class="dot" style="background:var(--good)"></i>vs random</span>
        <span><i class="dot" style="background:var(--p2)"></i>vs AlphaBeta</span>
        <span><i class="dot" style="background:#8b5cf6"></i><span id="leg-ref">vs reference</span></span>
        <span><i class="dot" style="background:var(--p1)"></i>gate (cand vs best)</span>
        <span><i class="dot" style="background:rgba(24,160,88,.35)"></i>accepted</span>
      </div>
    </div>
    <div class="card">
      <h2>Training loss</h2>
      <div class="chart" id="w-loss"><canvas id="c-loss" width="640" height="220"></canvas>
        <div class="vline"></div><div class="tip"></div></div>
      <div class="legend">
        <span><i class="dot" style="background:var(--p1)"></i>policy</span>
        <span><i class="dot" style="background:var(--p2)"></i>value</span>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="rowhead">
      <h2 id="right-title" style="margin:0">Latest self-play game</h2>
      <button id="mode-btn">&#9654; Play the best</button>
    </div>

    <div id="watch-view">
      <div id="board" class="board"></div>
      <div class="gamebar">
        <span id="g-move">&ndash;</span>
        <span id="g-winner"></span>
        <span><button id="g-toggle">pause</button></span>
      </div>
    </div>

    <div id="play-view" style="display:none">
      <div id="pboard" class="board"></div>
      <div class="gamebar">
        <span id="pmsg">&ndash;</span>
        <span><button id="newgame-btn">New game</button></span>
      </div>
      <div id="pstats" class="pstats"></div>
    </div>

    <div class="logwrap">
      <table class="log">
        <thead><tr><th>iter</th><th>elo</th><th>games</th><th>vs</th><th>gate</th><th>rnd</th><th>AB</th></tr></thead>
        <tbody id="logbody"></tbody>
      </table>
    </div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmtPct=v=>v==null||isNaN(v)?'–':Math.round(v*100)+'%';
function fmtDur(s){s=Math.floor(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60),ss=s%60;
  return h?`${h}h ${String(m).padStart(2,'0')}m`:`${m}m ${String(ss).padStart(2,'0')}s`;}

let ROWS=[], RATINGS={versions:[]}, RATED={};
const CHARTS={};   // canvasId -> {x0,x1,xmax}

// ---------- charts ----------
function draw(canvas, series, ymin, ymax, xmax, accepted, hlines){
  const ctx=canvas.getContext('2d'), W=canvas.width, H=canvas.height;
  const padL=38,padR=10,padT=10,padB=34;
  ctx.clearRect(0,0,W,H);
  const x0=padL,x1=W-padR,y0=H-padB,y1=padT;
  const X=t=>x0+(x1-x0)*(xmax<=1?0.5:(t-1)/(xmax-1||1));
  const Y=v=>y0+(y1-y0)*((v-ymin)/(ymax-ymin||1));
  CHARTS[canvas.id]={x0,x1,xmax};
  const digits=(ymax-ymin)>=20?0:1;
  // horizontal grid + y labels
  ctx.strokeStyle='#eef1f9';ctx.fillStyle='#9aa3bd';ctx.font='11px system-ui';ctx.lineWidth=1;ctx.textAlign='left';
  for(let i=0;i<=4;i++){const v=ymin+(ymax-ymin)*i/4,y=Y(v);
    ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x1,y);ctx.stroke();ctx.fillText(v.toFixed(digits),4,y+3);}
  // horizontal reference lines (e.g. reference-model Elo bars to beat)
  if(hlines)for(const h of hlines){if(h.y<ymin||h.y>ymax)continue;const y=Y(h.y);
    ctx.strokeStyle=h.color;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x1,y);ctx.stroke();
    ctx.setLineDash([]);ctx.fillStyle=h.color;ctx.textAlign='right';ctx.fillText(h.label,x1-2,y-3);ctx.textAlign='left';}
  // accepted-candidate markers (faint vertical lines)
  if(accepted){ctx.strokeStyle='rgba(24,160,88,.22)';
    for(const a of accepted){ctx.beginPath();ctx.moveTo(X(a),y0);ctx.lineTo(X(a),y1);ctx.stroke();}}
  // x-axis: iteration ticks
  ctx.textAlign='center';const nT=Math.min(8,Math.max(1,xmax));const seen=new Set();
  for(let i=0;i<=nT;i++){let t=Math.round(1+(xmax-1)*i/nT);if(t<1)t=1;
    if(seen.has(t))continue;seen.add(t);const x=X(t);
    ctx.strokeStyle='#f2f4fb';ctx.beginPath();ctx.moveTo(x,y0);ctx.lineTo(x,y1);ctx.stroke();
    ctx.fillStyle='#9aa3bd';ctx.fillText(t,x,y0+14);}
  ctx.fillStyle='#c2cbe0';ctx.fillText('iteration',(x0+x1)/2,H-2);ctx.textAlign='left';
  // series
  for(const s of series){
    const pts=s.pts.filter(p=>p.y!=null&&!isNaN(p.y));if(!pts.length)continue;
    ctx.setLineDash([]);ctx.strokeStyle=s.color;ctx.lineWidth=2.5;ctx.beginPath();
    pts.forEach((p,i)=>{const a=X(p.x),b=Y(p.y);i?ctx.lineTo(a,b):ctx.moveTo(a,b);});
    ctx.stroke();ctx.lineWidth=1;
    const last=pts[pts.length-1];ctx.fillStyle=s.color;ctx.beginPath();ctx.arc(X(last.x),Y(last.y),3.5,0,7);ctx.fill();
  }
}

function hookHover(boxId, canvasId, fields){
  const box=$(boxId), cv=$(canvasId), tip=box.querySelector('.tip'), vl=box.querySelector('.vline');
  cv.addEventListener('mousemove',e=>{
    const g=CHARTS[canvasId];if(!g||!ROWS.length)return;
    const rect=cv.getBoundingClientRect(), sx=cv.width/rect.width;
    const cx=(e.clientX-rect.left)*sx;
    let t=Math.round(1+(cx-g.x0)/((g.x1-g.x0)||1)*(g.xmax-1));
    t=Math.max(1,Math.min(g.xmax,t));
    const row=ROWS.find(r=>r.iter===t);if(!row){tip.style.opacity=0;vl.style.opacity=0;return;}
    const px=(g.x0+(g.x1-g.x0)*((t-1)/((g.xmax-1)||1)))/sx;
    vl.style.left=px+'px';vl.style.height=rect.height+'px';vl.style.opacity=1;
    let html='<div class="th">iteration '+t+'</div>';
    for(const f of fields()){const v=f.get(row);
      html+='<div class="r"><i class="dot" style="background:'+f.color+'"></i>'+f.label+': '+
            (v==null||isNaN(v)?'—':f.fmt(v))+'</div>';}
    tip.innerHTML=html;tip.style.left=px+'px';tip.style.top=Math.max(6,e.clientY-rect.top)+'px';tip.style.opacity=1;
  });
  cv.addEventListener('mouseleave',()=>{tip.style.opacity=0;vl.style.opacity=0;});
}

async function refreshMetrics(){
  let rows;try{rows=await(await fetch('/api/metrics')).json();}catch(e){return;}
  ROWS=rows;
  try{RATINGS=await(await fetch('/api/ratings')).json();}catch(e){}
  const byId={};(RATINGS.versions||[]).forEach(v=>{byId[v.id]=v;});RATED=byId;
  const xmax=Math.max(1,rows.length?rows[rows.length-1].iter:1);
  const accepted=rows.filter(r=>r.accepted).map(r=>r.iter);
  // elo chart (AlphaBeta = 0): current model + champion, with reference bars to beat
  const eloCur=rows.map(r=>({x:r.iter,y:r.elo}));
  const eloBest=rows.map(r=>({x:r.iter,y:r.best_elo}));
  const refs=(RATINGS.versions||[]).filter(v=>v.ref);
  const hlines=refs.map(v=>({y:v.elo,label:v.id.replace('ref:',''),color:'rgba(224,109,0,.7)'}));
  const evals=eloCur.concat(eloBest).map(p=>p.y).filter(v=>v!=null&&!isNaN(v))
                    .concat(refs.map(v=>v.elo)).concat([0]);
  let emin=Math.min(...evals),emax=Math.max(...evals);
  const epad=(emax-emin)*0.15||40;emin-=epad;emax+=epad;
  draw($('c-elo'),[{color:css('--good'),pts:eloCur},{color:css('--p1'),pts:eloBest}],
       emin,emax,xmax,accepted,hlines);
  const pol=rows.map(r=>({x:r.iter,y:r.policy_loss}));
  const val=rows.map(r=>({x:r.iter,y:r.value_loss}));
  const clean=pol.concat(val).map(p=>p.y).filter(v=>v!=null&&!isNaN(v));
  const lmax=Math.max(0.6,...clean);
  draw($('c-loss'),[{color:css('--p1'),pts:pol},{color:css('--p2'),pts:val}],0,lmax,xmax);
  const rnd=rows.map(r=>({x:r.iter,y:r.winrate_vs_random}));
  const ab=rows.map(r=>({x:r.iter,y:r.winrate_vs_alphabeta}));
  const ref=rows.map(r=>({x:r.iter,y:r.winrate_vs_reference}));
  const gate=rows.map(r=>({x:r.iter,y:r.cand_winrate_vs_best}));
  const refName=(rows.find(r=>r.reference_id)||{}).reference_id;
  if(refName)$('leg-ref').textContent='vs '+refName;
  const half=[{x:1,y:0.5},{x:xmax,y:0.5}];
  draw($('c-win'),[{color:'#dfe4ef',pts:half},{color:css('--good'),pts:rnd},
     {color:css('--p2'),pts:ab},{color:'#8b5cf6',pts:ref},{color:css('--p1'),pts:gate}],0,1,xmax,accepted);
  // counters
  const done=rows.length;
  const sumSec=rows.reduce((a,r)=>a+(r.iter_sec||0),0);
  if(done!==_rowCount){_rowCount=done;_elapsedBase=sumSec;_baseAt=performance.now();}
  $('s-speed').textContent=done?(sumSec/done).toFixed(0)+'s':'–';
  if(done){const last=rows[done-1];
    $('s-elo').textContent=last.best_elo!=null?Math.round(last.best_elo):'–';
    $('s-rand').textContent=fmtPct(last.winrate_vs_random);
    $('s-ab').textContent=fmtPct(last.winrate_vs_alphabeta);}
  renderTable(rows);
}

function renderTable(rows){
  let html='';
  for(let i=rows.length-1;i>=0&&i>rows.length-1-50;i--){
    const r=rows[i], gated=r.cand_winrate_vs_best!=null;
    const cls=gated?(r.accepted?'acc':'rej'):'';
    const tag=gated?(r.accepted?'<span class="tag acc">ACC</span>':'<span class="tag rej">rej</span>'):'';
    const gatev=gated?Math.round(r.cand_winrate_vs_best*100)+'% '+tag:'—';
    const live=RATED['v'+r.iter]||{};                     // latest Elo + game count
    const elo=live.elo!=null?Math.round(live.elo):(r.elo==null?'–':Math.round(r.elo));
    const gm=live.games!=null?live.games:'';
    const vs=r.version==null?'?':('v'+r.version);      // previous best it was gated against
    html+=`<tr class="${cls}"><td>#${r.iter}</td><td>${elo}</td><td>${gm}</td><td>${vs}</td><td>${gatev}</td>`+
          `<td>${fmtPct(r.winrate_vs_random)}</td><td>${fmtPct(r.winrate_vs_alphabeta)}</td></tr>`;
  }
  $('logbody').innerHTML=html;
}

// ---------- live counters ----------
let _rowCount=-1,_elapsedBase=0,_baseAt=performance.now();
function tickElapsed(){if(_rowCount<0){return;}
  $('s-elapsed').textContent=fmtDur(_elapsedBase+(performance.now()-_baseAt)/1000);}

async function refreshStatus(){
  let st;try{st=await(await fetch('/api/status')).json();}catch(e){
    $('phase').textContent='offline';$('phase').classList.add('offline');return;}
  $('phase').classList.remove('offline');
  const ph=$('phase');ph.textContent=st.phase||'idle';ph.classList.toggle('done',st.phase==='done');
  if(st.iter!=null)$('s-iter').textContent=st.iter;
  if(st.buffer!=null)$('s-buf').textContent=st.buffer.toLocaleString();
  if(st.phase==='self-play'&&st.game){$('s-extra').textContent=st.game+'/'+st.games;$('s-extra-k').textContent='self-play game';}
  else{$('s-extra').textContent=st.phase||'–';$('s-extra-k').textContent='phase';}
}

// ---------- board replay ----------
let frames=[],fi=0,gameIter=null,playing=true,size=4;
function renderFrame(){
  if(!frames.length)return;
  const f=frames[fi%frames.length], b=$('board');
  b.style.gridTemplateColumns=`repeat(${size},1fr)`;
  let html='';
  for(let y=0;y<size;y++)for(let x=0;x<size;x++){
    const owner=f.player[y][x],cnt=f.board[y][x];
    const isLast=f.move&&f.move[0]===x&&f.move[1]===y;
    html+=`<div class="cell${owner===1?' p1':owner===2?' p2':''}${isLast?' last':''}">${cnt||''}</div>`;
  }
  b.innerHTML=html;
  $('g-move').textContent=`move ${fi%frames.length} / ${frames.length-1}`;
}
async function refreshGame(){
  let g;try{g=await(await fetch('/api/game')).json();}catch(e){return;}
  if(!g.frames.length)return;
  if(g.iter!==gameIter){gameIter=g.iter;frames=g.frames;size=g.size;fi=0;
    const w=$('g-winner');
    if(g.winner){w.textContent='winner: P'+g.winner;w.className='winner p'+g.winner;}
    else{w.textContent='draw';w.className='winner';}}
}
function tick(){if(playing&&frames.length){fi=(fi+1)%frames.length;renderFrame();}}
$('g-toggle').onclick=()=>{playing=!playing;$('g-toggle').textContent=playing?'pause':'play';};

// ---------- play the best model ----------
let mode='watch',playVer=null,playSize=4,humanTurn=false;
function showViews(){
  $('watch-view').style.display=mode==='watch'?'':'none';
  $('play-view').style.display=mode==='play'?'':'none';
  $('right-title').textContent=mode==='watch'?'Latest self-play game':('Play vs best'+(playVer!=null?' · v'+playVer:''));
  $('mode-btn').innerHTML=mode==='watch'?'&#9654; Play the best':'&#8249; Back to watching';
}
$('mode-btn').onclick=async()=>{if(mode==='watch'){await startPlay();}else{mode='watch';showViews();}};
$('newgame-btn').onclick=startPlay;

async function startPlay(){
  mode='play';showViews();$('pstats').innerHTML='';$('pmsg').textContent='Loading current best model…';
  let r;try{r=await(await fetch('/api/play/new',{method:'POST'})).json();}catch(e){$('pmsg').textContent='error';return;}
  if(!r.ok){$('pmsg').textContent=r.error||'error';return;}
  playVer=r.version;playSize=r.state.size;humanTurn=true;showViews();
  renderPlay(r.state);
  $('pmsg').innerHTML='Your move — you are <b style="color:var(--p1)">blue (P1)</b>; AI is orange.';
}
function renderPlay(state){
  const b=$('pboard');b.style.gridTemplateColumns=`repeat(${state.size},1fr)`;
  const legal=new Set((state.legal||[]).map(m=>m[0]+','+m[1]));
  const canClick=humanTurn&&state.current===1;
  let html='';
  for(let y=0;y<state.size;y++)for(let x=0;x<state.size;x++){
    const owner=state.player[y][x],cnt=state.board[y][x],lg=canClick&&legal.has(x+','+y);
    html+=`<div class="cell${owner===1?' p1':owner===2?' p2':''}${lg?' legal':''}" data-x="${x}" data-y="${y}">${cnt||''}</div>`;
  }
  b.innerHTML=html;
  if(canClick)b.querySelectorAll('.cell.legal').forEach(c=>c.onclick=()=>humanMove(+c.dataset.x,+c.dataset.y));
}
function endMsg(st){
  if(!st.over)return false;
  const w=st.winner;
  $('pmsg').innerHTML=w===1?'<b style="color:var(--p1)">You win!</b> ('+st.reason+')'
    :w===2?'<b style="color:var(--p2)">AlphaZero wins.</b> ('+st.reason+')':'Draw. ('+st.reason+')';
  humanTurn=false;return true;
}
async function humanMove(x,y){
  if(!humanTurn)return;humanTurn=false;
  let r;try{r=await(await fetch('/api/play/human',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({x,y})})).json();}catch(e){humanTurn=true;return;}
  if(!r.ok){humanTurn=true;return;}
  renderPlay(r.state);
  if(endMsg(r.status))return;
  $('pmsg').innerHTML='AlphaZero is thinking… <span style="opacity:.6">(5s)</span>';
  let a;try{a=await(await fetch('/api/play/ai',{method:'POST'})).json();}catch(e){$('pmsg').textContent='AI error';return;}
  if(!a.ok){$('pmsg').textContent=a.error||'AI error';return;}
  humanTurn=!a.status.over;
  renderPlay(a.state);markAiMove(a.ai_move);showStats(a.stats);
  if(endMsg(a.status))return;
  $('pmsg').innerHTML='Your move.';
}
function markAiMove(mv){if(!mv)return;const c=$('pboard').querySelector(`.cell[data-x="${mv[0]}"][data-y="${mv[1]}"]`);if(c)c.classList.add('aimove');}
function showStats(s){
  if(!s){$('pstats').innerHTML='';return;}
  const pv=s.pv.map(m=>`(${m[0]},${m[1]})`).join(' → ');
  const rows=s.top.map(t=>`<tr><td>(${t.move[0]},${t.move[1]})</td><td>${t.visits}</td><td>${t.q>=0?'+':''}${t.q}</td><td>${Math.round(t.prior*100)}%</td></tr>`).join('');
  const n=playSize,pmax=Math.max(...s.net_policy,0.0001);
  let heat='<div class="heat" style="grid-template-columns:repeat('+n+',1fr)">';
  for(let i=0;i<s.net_policy.length;i++){const p=s.net_policy[i],a=p/pmax;
    const bg=p>0?`rgba(45,125,255,${(0.12+0.8*a).toFixed(2)})`:'#f4f6fb';
    heat+=`<div class="hc" style="background:${bg}">${p>0.005?Math.round(p*100):''}</div>`;}
  heat+='</div>';
  const sv=Math.round(s.win_prob*100),nv=Math.round(s.net_win_prob*100);
  $('pstats').innerHTML=
    `<div class="sline">After <b>${s.sims}</b> simulations, AlphaZero played <b>(${s.move[0]},${s.move[1]})</b> — it estimates a <b>${sv}% win</b>.</div>`+
    `<div class="sline">PV (expected line): <span class="pv">${pv||'—'}</span></div>`+
    `<div class="heads">`+
      `<div class="mini"><h4>Policy head</h4>${heat}</div>`+
      `<div class="mini"><h4>Value head</h4><div class="vbar"><span class="mk" style="left:${nv}%"></span></div>`+
        `<div class="sline" style="margin-top:6px">net value <b>${s.net_value>=0?'+':''}${s.net_value}</b> (&#8776; ${nv}% win, pre-search)</div></div>`+
    `</div>`+
    `<table class="log" style="margin-top:10px"><thead><tr><th>move</th><th>visits</th><th>Q</th><th>prior</th></tr></thead><tbody>${rows}</tbody></table>`;
}

hookHover('w-elo','c-elo',()=>[
  {label:'current model',color:css('--good'),get:r=>(RATED['v'+r.iter]||{}).elo??r.elo,fmt:v=>v.toFixed(0)},
  {label:'champion',color:css('--p1'),get:r=>r.best_elo,fmt:v=>v.toFixed(0)},
  {label:'games',color:'#9aa3bd',get:r=>(RATED['v'+r.iter]||{}).games,fmt:v=>String(v)}]);
hookHover('w-loss','c-loss',()=>[
  {label:'policy',color:css('--p1'),get:r=>r.policy_loss,fmt:v=>v.toFixed(3)},
  {label:'value',color:css('--p2'),get:r=>r.value_loss,fmt:v=>v.toFixed(3)}]);
hookHover('w-win','c-win',()=>[
  {label:'vs random',color:css('--good'),get:r=>r.winrate_vs_random,fmt:fmtPct},
  {label:'vs AlphaBeta',color:css('--p2'),get:r=>r.winrate_vs_alphabeta,fmt:fmtPct},
  {label:'vs reference',color:'#8b5cf6',get:r=>r.winrate_vs_reference,fmt:fmtPct},
  {label:'gate',color:css('--p1'),get:r=>r.cand_winrate_vs_best,fmt:fmtPct}]);

refreshMetrics();refreshStatus();refreshGame().then(renderFrame);
setInterval(refreshMetrics,2500);
setInterval(refreshStatus,1500);
setInterval(refreshGame,4000);
setInterval(tick,450);
setInterval(tickElapsed,1000);
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="StackIt AlphaZero training dashboard")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"Dashboard: http://{args.host}:{args.port}  (reading {CKPT}/)")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
