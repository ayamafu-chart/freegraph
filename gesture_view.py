"""FreeGraph: グラフ画像を表示し、指の操作 (ドラッグ・ピンチ・回転) を Python 側へ伝える Streamlit 部品。

Streamlit 標準の画像表示では指の操作を受け取れないため、小さな HTML/JavaScript の部品を使います。
HTML は下の文字列としてこのファイルに入っており、起動時に一時フォルダへ書き出して使います (別ファイルの追加は不要)。

Python へ返す値 (どれも {"id": 重複しない文字列, ...}):
  {"type": "view", "s": 拡大率, "tx": 横移動(画像の幅に対する割合), "ty": 縦移動(画像の高さに対する割合)}   2D: ドラッグ/ピンチ
  {"type": "zoom", "s": 拡大率}                                                                  3D: ピンチ
  {"type": "rot",  "dx": 横移動(画像の幅に対する割合), "dy": 縦移動(同)}                          3D: 1本指ドラッグ (操作中も少しずつ送る)
"""
import os
import tempfile

import streamlit.components.v1 as components

_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<style>
  html, body {margin:0; padding:0; background:transparent; overflow:hidden;}
  #wrap {position:relative; width:100%; overflow:hidden; box-sizing:border-box; background:#fff;
         border:1px solid #bbb; touch-action:none; user-select:none; -webkit-user-select:none;
         -webkit-touch-callout:none; -webkit-tap-highlight-color:transparent;}
  #img {width:100%; display:block; transform-origin:0 0; will-change:transform; pointer-events:none;
        -webkit-user-drag:none;}
  #busy {position:absolute; right:8px; top:6px; font:12px sans-serif; color:#777; display:none;
         background:rgba(255,255,255,.8); padding:1px 6px; border-radius:3px;}
</style></head>
<body>
<div id="wrap"><img id="img" alt="" draggable="false"><div id="busy">更新中…</div></div>
<script>
(function () {
  var wrap = document.getElementById('wrap'), img = document.getElementById('img'), busy = document.getElementById('busy');
  var args = {mode: '2d', enabled: true}, lastRev = null, waiting = false, waitTimer = null;
  var M = {s: 1, tx: 0, ty: 0};            // 表示中の画像にかけている変換: p' = s*p + t (ピクセル)
  var pts = {};                            // 押されている指 {pointerId: {x, y}}
  var base = null, startPts = null, pinched = false, rotLast = null, seq = 0, wheelTimer = null;

  function post(type, data) {
    var m = {isStreamlitMessage: true, type: type};
    for (var k in (data || {})) m[k] = data[k];
    window.parent.postMessage(m, '*');
  }
  function count() { return Object.keys(pts).length; }
  function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
  function applyT() { img.style.transform = 'translate(' + M.tx + 'px,' + M.ty + 'px) scale(' + M.s + ')'; }
  function resetT() { M = {s: 1, tx: 0, ty: 0}; applyT(); }
  function setHeight() { post('streamlit:setFrameHeight', {height: Math.ceil(wrap.getBoundingClientRect().height) + 2}); }
  function setWait(b, quiet) {
    waiting = b; busy.style.display = (b && !quiet) ? 'block' : 'none'; clearTimeout(waitTimer);
    if (b) waitTimer = setTimeout(function () { waiting = false; busy.style.display = 'none'; resetT(); }, 6000);
  }
  function send(v, quiet) {
    v.id = Date.now() + '-' + (++seq);
    post('streamlit:setComponentValue', {value: v, dataType: 'json'});
    setWait(true, quiet);
  }
  function rel(e) { var r = wrap.getBoundingClientRect(); return {x: e.clientX - r.left, y: e.clientY - r.top}; }
  function center() { var r = wrap.getBoundingClientRect(); return {x: r.width / 2, y: r.height / 2}; }
  function snapshot() { var o = {}; for (var k in pts) o[k] = {x: pts[k].x, y: pts[k].y}; return o; }
  function rebase() { base = {s: M.s, tx: M.tx, ty: M.ty}; startPts = snapshot(); }

  // 指の動きから、画像にかける変換を計算して見た目に反映する
  function update() {
    var ids = Object.keys(pts).filter(function (k) { return startPts && startPts[k]; });
    if (!ids.length) return;
    var sg = 1, cA, cB;
    if (ids.length >= 2) {
      var a0 = startPts[ids[0]], a1 = startPts[ids[1]], b0 = pts[ids[0]], b1 = pts[ids[1]];
      var dA = Math.hypot(a1.x - a0.x, a1.y - a0.y) || 1, dB = Math.hypot(b1.x - b0.x, b1.y - b0.y);
      sg = dB / dA;
      cA = {x: (a0.x + a1.x) / 2, y: (a0.y + a1.y) / 2}; cB = {x: (b0.x + b1.x) / 2, y: (b0.y + b1.y) / 2};
    } else { cA = startPts[ids[0]]; cB = pts[ids[0]]; }
    if (args.mode === '3d') { cA = cB = center(); }          // 3D: 移動はせず、中心を基準に拡大縮小だけ
    var s = clamp(sg * base.s, 0.05, 20), k = s / base.s;
    M = {s: s, tx: k * (base.tx - cA.x) + cB.x, ty: k * (base.ty - cA.y) + cB.y};
    applyT();
  }

  function commitView() {
    var r = wrap.getBoundingClientRect();
    if (Math.abs(M.s - 1) < 0.005 && Math.abs(M.tx) < 2 && Math.abs(M.ty) < 2) { resetT(); return; }
    if (args.mode === '3d') send({type: 'zoom', s: M.s});
    else send({type: 'view', s: M.s, tx: M.tx / r.width, ty: M.ty / r.height});
  }

  // 3D: 1本指ドラッグで回転。サーバーの描画が終わるたびに、それまでの動き分を送る (操作中も絵が追従する)
  function rotMove() {
    if (waiting || !rotLast) return;
    var k = Object.keys(pts); if (k.length !== 1) return;
    var cur = pts[k[0]], dx = cur.x - rotLast.x, dy = cur.y - rotLast.y;
    if (Math.abs(dx) + Math.abs(dy) < 3) return;
    var w = wrap.getBoundingClientRect().width;
    rotLast = {x: cur.x, y: cur.y};
    send({type: 'rot', dx: dx / w, dy: dy / w}, true);
  }

  wrap.addEventListener('pointerdown', function (e) {
    if (!args.enabled || waiting) return;
    try { wrap.setPointerCapture(e.pointerId); } catch (_) {}
    pts[e.pointerId] = rel(e);
    if (count() === 1) { pinched = false; rotLast = rel(e); } else { pinched = true; }
    rebase(); e.preventDefault();
  });
  wrap.addEventListener('pointermove', function (e) {
    if (!pts[e.pointerId]) return;
    pts[e.pointerId] = rel(e);
    if (args.mode === '3d' && count() === 1 && !pinched) { rotMove(); return; }
    update(); e.preventDefault();
  });
  function up(e) {
    if (!pts[e.pointerId]) return;
    if (args.mode === '3d' && count() === 1 && !pinched) rotMove();
    var was3dRot = (args.mode === '3d' && !pinched);
    delete pts[e.pointerId];
    if (count() > 0) { rebase(); return; }          // まだ指が残っている → 基準を取り直して続ける
    if (was3dRot) return;
    commitView();
  }
  wrap.addEventListener('pointerup', up);
  wrap.addEventListener('pointercancel', up);

  // PCのマウスホイール: 拡大縮小
  wrap.addEventListener('wheel', function (e) {
    if (!args.enabled) return;
    e.preventDefault(); if (waiting) return;
    var p = args.mode === '3d' ? center() : rel(e), f = Math.exp(-e.deltaY * 0.0015);
    var s = clamp(M.s * f, 0.05, 20), k = s / M.s;
    M = {s: s, tx: k * (M.tx - p.x) + p.x, ty: k * (M.ty - p.y) + p.y};
    applyT(); clearTimeout(wheelTimer);
    wheelTimer = setTimeout(function () { if (count() === 0) commitView(); }, 220);
  }, {passive: false});

  // サーバーから新しい画像が届いた
  function onRender(a) {
    args = a || args;
    wrap.style.touchAction = args.enabled ? 'none' : 'pan-y';
    if (args.rev === lastRev && img.src && img.getAttribute('data-has') === '1') return;
    lastRev = args.rev;
    var ni = new Image();
    ni.onload = function () {
      img.src = args.img; img.setAttribute('data-has', '1'); resetT(); setWait(false); setHeight();
      if (startPts) rebase();                              // 操作の途中なら、新しい画像を基準に続ける
      if (args.mode === '3d' && count() === 1 && !pinched) {
        var k = Object.keys(pts)[0]; rotMove();            // 待っている間に動いた分を送る
      }
    };
    ni.src = args.img;
  }
  window.addEventListener('message', function (ev) {
    if (ev.data && ev.data.type === 'streamlit:render') onRender(ev.data.args);
  });
  if (window.ResizeObserver) new ResizeObserver(setHeight).observe(wrap);
  window.addEventListener('resize', setHeight);
  post('streamlit:componentReady', {apiVersion: 1});
  setHeight();
})();
</script></body></html>
"""


def _setup_dir():
    for base in (tempfile.gettempdir(), os.path.dirname(os.path.abspath(__file__))):
        try:
            d = os.path.join(base, "freegraph_gesture_component")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
                f.write(_HTML)
            return d
        except OSError:
            continue
    raise RuntimeError("部品用のフォルダを作れませんでした")


_component = components.declare_component("freegraph_gesture", path=_setup_dir())


def graph_view(img_data_url, mode, rev, enabled=True, key=None):
    """グラフ画像(PNGのdata URL)を表示し、指の操作があればその内容(dict)を返す。なければ None。
    mode: "2d" (ドラッグ+ピンチ) / "3d" (回転+ピンチ)。rev は描画のたびに変わる番号。"""
    return _component(img=img_data_url, mode=mode, rev=rev, enabled=enabled, key=key, default=None)
