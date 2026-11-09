"""MathPlot Studio (Web版) - Streamlit から呼び出す描画エンジン。

mathplot_core.py（数式の解析・描画ロジック）を、PySide6(GUI)なしで使うための橋渡しです。
このファイルは streamlit に依存しないので、パソコン上で単体テストもできます。
"""
import os
os.environ.setdefault("MPS_NOGUI", "1")      # mathplot_core の GUI(PySide6) 部分を読み込まない
import io, re, json, hashlib, threading
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from types import SimpleNamespace
import mathplot_core as core

LOCK = threading.RLock()        # core は共有のグローバル状態を持つので、同時アクセスを直列化する
SPACES = core.SPACES
SPACE_NAMES = list(SPACES)
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
AX_BOX = (0.055, 0.065, 0.94, 0.93)      # 2Dグラフの描画領域 (図全体に対する 左, 下, 幅, 高さ)

# ---------------- キーパッド ----------------
DEFAULT_KEYS = [
    {"name": "基本", "keys": [["7", "7"], ["8", "8"], ["9", "9"], ["÷", "/"], ["(", "("], [")", ")"], ["x", "x"],
                           ["4", "4"], ["5", "5"], ["6", "6"], ["×", "*"], ["x²", "^2"], ["xⁿ", "^(|)"], ["y", "y"],
                           ["1", "1"], ["2", "2"], ["3", "3"], ["−", "-"], ["√", "sqrt(|)"], ["|x|", "abs(|)"], ["z", "z"],
                           ["0", "0"], [".", "."], [",", ","], ["+", "+"], ["a/b", "(|)/()"], ["=", "="], ["t", "t"]]},
    {"name": "関数", "keys": [[a, f"{a}(|)"] for a in ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                                                        "asinh", "acosh", "atanh", "ln", "log", "exp", "sgn", "floor", "conj", "re", "im", "arg")] + [["logₐ", "log(|,10)"], ["det", "det(|,)"]]},
    {"name": "定数・変数", "keys": [["π", "pi"], ["e", "e"], ["i", "i"], ["θ", "th"], ["r", "r"], ["s", "s"], ["t", "t"],
                               ["n", "n"], ["m", "m"], ["k", "k"], ["u", "u"]]},
    {"name": "微積・和", "keys": [["Σ", "sum(|,k,1,10)"], ["Π", "prod(|,k,1,10)"], ["∫", "integral(|,u,0,1)"], ["d/dx", "diff(|,x)"]]},
    {"name": "式・条件", "keys": [["( , )", "(|,)"], ["vec", "vec(|)"], ["field", "field(|)"], ["basis", "basis(|)"], ["{ }", "{|}"],
                              ["<", "<"], [">", ">"], ["≤", "<="], ["≥", ">="], ["and", " and "], ["or", " or "]]},
    {"name": "定義・ベクトル", "keys": [["A( , )", "A(|,)"], ["f(x)=", "f(x)="], ["α=", "α="], ["a⃗=vec", "a⃗=vec(|)"], ["det", "det(|,)"],
                                  ["dot", "dot(|,)"], ["cross", "cross(|,)"], ["norm", "norm(|)"], ["w=", "w="], ["|z|", "|z|"], ["α", "α"], ["β", "β"]]},
    {"name": "行列・ベクトル", "keys": [["u·v", "dot(|,)"], ["u×v", "cross(|,)"], ["‖u‖", "norm(|)"], ["2×2", "mat2(|,,,)"], ["3×3", "mat3(|,,,,,,,,)"],
                                  ["M⁻¹", "inv(|)"], ["det", "det(|)"], ["Mᵀ", "transp(|)"], ["M₂=", "M=mat2(|,,,)"], ["M₃=", "M=mat3(|,,,,,,,,)"],
                                  ["a⃗=", "a⃗=vec(|,)"], ["A( , )", "A(|,)"]]},
    {"name": "特殊関数", "keys": [["ζ", "zeta(|)"], ["Γ", "gamma(|)"], ["lnΓ", "loggamma(|)"], ["ψ", "digamma(|)"], ["B(a,b)", "beta(|,)"], ["n!", "factorial(|)"], ["C(n,k)", "binomial(|,)"],
                                ["erf", "erf(|)"], ["erfc", "erfc(|)"], ["erfi", "erfi(|)"], ["Jν", "besselj(|,)"], ["Yν", "bessely(|,)"], ["Iν", "besseli(|,)"], ["Kν", "besselk(|,)"],
                                ["Ai", "airyai(|)"], ["Bi", "airybi(|)"], ["W", "lambertw(|)"], ["Si", "Si(|)"], ["Ci", "Ci(|)"], ["Ei", "Ei(|)"], ["li", "li(|)"],
                                ["Eₙ", "expint(|,)"], ["K(m)", "ellipk(|)"], ["E(m)", "ellipe(|)"], ["Liₛ", "polylog(|,)"], ["η", "eta(|)"], ["π(x)", "primepi(|)"], ["φ(n)", "totient(|)"], ["mod", " % "]]}]


KEYS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mathplot_studio_keys.json")
KEYS_VER = 3
NEWPAGES = {"行列・ベクトル", "特殊関数"}          # 旧バージョンのキー設定には、新しい既定ページを足す (デスクトップ版と同じ動き)


def parse_keypages(raw):
    """キー設定JSON(デスクトップ版と同じ形式)を検証してページ一覧にする。不正なら ValueError"""
    data, ver = (raw["pages"], raw.get("v", 1)) if isinstance(raw, dict) else (raw, 1)
    ok = isinstance(data, list) and data and all(
        isinstance(p, dict) and isinstance(p.get("name"), str) and isinstance(p.get("keys"), list)
        and all(isinstance(k, (list, tuple)) and len(k) == 2 and all(isinstance(x, str) for x in k) for k in p["keys"])
        for p in data)
    if not ok:
        raise ValueError("キー設定の形式が正しくありません")
    data = json.loads(json.dumps(data))
    if ver < KEYS_VER:
        have = {p["name"] for p in data}
        data += [json.loads(json.dumps(p)) for p in DEFAULT_KEYS if p["name"] in NEWPAGES and p["name"] not in have]
    return data


_keys_cache = {}


def load_keypages():
    """mathplot_studio_keys.json があればそれを、無ければ(または壊れていれば)既定のキーパッドを返す。戻り値: (ページ一覧, 説明)"""
    try:
        mt = os.path.getmtime(KEYS_FILE)
    except OSError:
        return json.loads(json.dumps(DEFAULT_KEYS)), "既定のキーパッド"
    if _keys_cache.get("mt") != mt:
        try:
            with open(KEYS_FILE, encoding="utf-8") as f:
                _keys_cache.update(mt=mt, pages=parse_keypages(json.load(f)), msg="mathplot_studio_keys.json を使用中")
        except Exception as ex:
            _keys_cache.update(mt=mt, pages=json.loads(json.dumps(DEFAULT_KEYS)), msg=f"mathplot_studio_keys.json を読めないため既定を使用中 ({ex})")
    return _keys_cache["pages"], _keys_cache["msg"]


def keypages(base, param_names):
    """キーパッドのページ一覧。末尾に現在のパラメータ用ページを自動追加。"""
    pages = [dict(name=p["name"], keys=[k for k in p["keys"] if k[1] != "(|)/()"]) for p in base]   # a/b キー(カーソル位置が必要)は Web では隠す
    return pages + [dict(name="パラメータ", keys=[[n, n] for n in param_names])]


def key_text(text):
    """キー1つで末尾に追加する文字列。| (カーソル位置の印) は Web では使えないので、手前までを入れる (閉じ括弧は自動補完)"""
    if text.count("|") == 1:
        head, tail = text.split("|")
        return head
    return text


def balance(s):
    """閉じ忘れの ( と { を末尾で自動的に閉じる"""
    stack = []
    for ch in s:
        if ch in "({": stack.append(")" if ch == "(" else "}")
        elif ch in ")}" and stack and stack[-1] == ch: stack.pop()
    return s + "".join(reversed(stack))


# ---------------- 定数・パラメータ ----------------
def set_consts(text):
    core.CONSTS.clear()
    for part in re.split(r"[,;]", text or ""):
        if "=" in part:
            k, v = part.split("=", 1)
            try: core.CONSTS[k.strip()] = core.parse_expr(v.strip().replace("π", "pi"), local_dict=core.LOC, transformations=core.TF)
            except Exception: pass


def param_range(p):
    """パラメータの範囲 (最小, 最大)。数値や π などの定数式を受け付ける。不正なら (-5, 5)"""
    with LOCK:
        try: a, b = core.num_of(p["lo"]), core.num_of(p["hi"])
        except Exception: a, b = -5.0, 5.0
    a, b = min(a, b), max(a, b)
    return (a, b) if b - a > 1e-9 else (a, a + 1.0)


def sync_params(params):
    """パラメータ名と現在値を式の評価エンジンへ反映。戻り値: 有効な名前の一覧 (t は除く)"""
    names, seen = [], {"t"}
    for p in params:
        n = p["name"].strip()
        ok = p.get("fixed") or bool(re.fullmatch(r"[^\W\d]\w*", n) and n not in seen and not core.param_reserved(n))
        if ok:
            seen.add(n); core.PVAL[n] = float(p["val"])
            if not p.get("fixed"): names.append(n)
    core.set_params(names)
    return names


def prepare(state):
    names = sync_params(state["params"])
    set_consts(state["consts"])
    errs = core.build_defs([balance(r["expr"]) for r in state["rows"]], SPACES[state["space"]]["key"])
    return names, errs


def analyze(state):
    """各行のTeXプレビューと入力エラーを返す。戻り値: (有効なパラメータ名, [(tex or None, エラー or None), ...])"""
    with LOCK:
        names, derrs = prepare(state)
        out = []
        for i, r in enumerate(state["rows"]):
            txt = balance(r["expr"]); tex = err = None
            if txt.strip():
                try: tex = core.tex_item(txt)
                except Exception as ex: err = f"入力エラー: {ex}"
                if derrs.get(i): err = "定義エラー: " + derrs[i]
            out.append((tex, err))
        return names, out


# ---------------- 描画 ----------------
_geo = {}          # 3D: 直近に描いた図。視点(elev/azim)だけが違うなら、データの計算をやり直さず視点だけ変えて保存する


def _save(fig, dpi, jpeg):
    buf = io.BytesIO()
    if jpeg: fig.savefig(buf, format="jpeg", dpi=dpi, pil_kwargs={"quality": 72})       # 軽い下書き用
    else: fig.savefig(buf, format="png", dpi=dpi)
    return buf.getvalue()


def _finish3d(fig, ax, errs, L, asp, state, dpi, jpeg):
    ax.view_init(state.get("elev", 25), state.get("azim", -60))
    core.fit3d(ax, fig, L, asp, 0.96)
    return _save(fig, dpi, jpeg), list(errs), L


def _render(state, w_in, h_in, dpi, jpeg):
    """グラフを画像のバイト列にして返す。戻り値: (画像, 行ごとのエラー文字列のリスト, 実際に使った軸範囲)"""
    with LOCK:
        S = SPACES[state["space"]]; dim = S["dim"]; tz = S.get("tz"); gk = None
        if dim == 3:                                      # 視点以外が同じなら、前回の図を使い回す
            gk = hashlib.md5(json.dumps([{k: v for k, v in state.items() if k not in ("elev", "azim")}, w_in, h_in],
                                        sort_keys=True, default=str).encode("utf-8")).hexdigest()
            if _geo.get("key") == gk:
                return _finish3d(*_geo["v"], state, dpi, jpeg)
        names, derrs = prepare(state)
        params = state["params"]; allp = ["t"] + names
        tname = state.get("axp", "t") if tz else "t"
        if tname not in allp: tname = "t"
        core.AX[0] = tname
        prow = next((p for p in params if p["name"].strip() == tname), params[0])
        t = float(prow["val"]); tr = param_range(prow)
        L = [float(v) for v in state["lim"]]
        if tz: L[4], L[5] = tr
        fig = Figure(figsize=(w_in, h_in), dpi=dpi, facecolor="white")
        W_, H_ = fig.bbox.width, fig.bbox.height
        if dim == 2 and state.get("equal", True):       # 縦横を等倍に
            w, h = W_ * AX_BOX[2], H_ * AX_BOX[3]
            u = max((L[1] - L[0]) / w, (L[3] - L[2]) / h); cx, cy = (L[0] + L[1]) / 2, (L[2] + L[3]) / 2
            L[:4] = [cx - u * w / 2, cx + u * w / 2, cy - u * h / 2, cy + u * h / 2]
        R = max(abs(v) for v in (L if dim == 3 and not tz else L[:4]))
        ax = fig.add_axes(list(AX_BOX) if dim == 2 else [0, 0, 1, 1], projection="3d" if dim == 3 else None)
        if dim == 3: getattr(ax, "disable_mouse_rotation", lambda: None)()
        ctx = SimpleNamespace(R=R, t=t, tr=tr, cond="", lw=2.0, maps=[])
        rows = state["rows"]
        if S["key"] == "cx":                              # w=f(z) の行を集める
            for r in rows:
                mm = re.match(core.MAP_RE, core.split_cond(balance(r["expr"]))[0])
                if mm and r["on"]:
                    try: ctx.maps.append(core.compile_expr(mm.group(1))[2])
                    except Exception: pass
        errs = [None] * len(rows)
        for i, r in enumerate(rows):
            txt = balance(r["expr"])
            if derrs.get(i): errs[i] = "定義エラー: " + derrs[i]; continue
            if not r["on"] or not txt.strip(): continue
            try:
                ctx.lw = float(r["lw"])
                with matplotlib.rc_context({"lines.linewidth": ctx.lw}):
                    core.draw_item(ax, txt, S, ctx, r["color"])
            except Exception as ex:
                errs[i] = f"描画エラー: {ex}"
        if S.get("polar"):                                # 極座標グリッド
            th_ = np.linspace(0, 2 * np.pi, 200)
            for k in range(1, 40): ax.plot(k * R / 5 * np.cos(th_), k * R / 5 * np.sin(th_), color="gray", lw=0.5, alpha=0.4)
            for a_ in np.arange(0, 2 * np.pi, np.pi / 6):
                ax.plot([0, 20 * R * np.cos(a_)], [0, 20 * R * np.sin(a_)], color="gray", lw=0.5, alpha=0.4)
        lb = list(S["labels"])
        if tz: lb[2] = tname
        if dim == 2:
            core.axes2d(ax, L, lb, max(4, int(round(10 * H_ / max(W_, 1)))))
        else:
            ax.set_xlim3d(L[0], L[1]); ax.set_ylim3d(L[2], L[3]); ax.set_zlim3d(L[4], L[5]); core.axes3d(ax, L, lb)
            kx = min(2.2, max(1.0, W_ / max(H_, 1))) ** 0.8
            asp = [L[1] - L[0], L[3] - L[2], L[5] - L[4]] if state.get("equal", True) else [kx, kx, 1]
            asp = tuple(v / max(asp) for v in asp)
            _geo.clear(); _geo.update(key=gk, v=(fig, ax, errs, L, asp))
            return _finish3d(fig, ax, errs, L, asp, state, dpi, jpeg)
        return _save(fig, dpi, jpeg), errs, L


_memo = OrderedDict()          # 直近の描画結果。同じ内容の再描画を省く (画面の再実行のたびに描き直さないため)


def render(state, w_in=8.0, h_in=7.0, dpi=130, draft=False):
    """グラフを描く。draft=True なら回転中・再生中向けの軽い画像 (低解像度のJPEG)。
    戻り値: (画像バイト列, 行ごとのエラー, 実際に使った軸範囲, 画像の種類 "png"/"jpeg", 内容を表すキー)"""
    jpeg = bool(draft)
    if draft: dpi = min(dpi, 72)
    key = hashlib.md5(json.dumps([state, w_in, h_in, dpi, jpeg], sort_keys=True, default=str).encode("utf-8")).hexdigest()
    with LOCK:
        if key in _memo:
            _memo.move_to_end(key); r = _memo[key]
        else:
            r = _render(state, w_in, h_in, dpi, jpeg)
            _memo[key] = r
            while len(_memo) > 6: _memo.popitem(last=False)
    return r[0], r[1], r[2], ("jpeg" if jpeg else "png"), key


# ---------------- 指の操作 (ジェスチャー) の計算 ----------------
def view_transform_2d(L, s, tx, ty):
    """2D: 表示中の画像に「拡大率 s・移動 (tx, ty) [画像の幅/高さに対する割合]」を掛けた見え方になるよう、軸範囲 L=[x0,x1,y0,y1,..] を更新して返す。
    画像上の点 p は p' = s*p + t へ動く。新しい画面の位置 q には、元の画面の p=(q-t)/s にあったデータが来る。"""
    s = min(max(float(s), 0.02), 50.0)
    ax0, ay0, aw, ah = AX_BOX; top = 1 - ay0 - ah
    x0, x1, y0, y1 = L[:4]; sx, sy = x1 - x0, y1 - y0

    def dx(q): return x0 + (((q - tx) / s - ax0) / aw) * sx
    def dy(q): return y1 - (((q - ty) / s - top) / ah) * sy
    nx0, nx1 = dx(ax0), dx(ax0 + aw)
    ny1, ny0 = dy(top), dy(top + ah)
    out = list(L); out[:4] = [nx0, nx1, ny0, ny1]
    return out


def zoom_lims(L, s, n):
    """各軸の中心を固定したまま、拡大率 s で拡大 (s>1) / 縮小 (s<1)。n = 対象の軸の数"""
    s = min(max(float(s), 0.02), 50.0); out = list(L)
    for i in range(n):
        lo, hi = L[2 * i], L[2 * i + 1]; c, w = (lo + hi) / 2, (hi - lo) / s
        out[2 * i], out[2 * i + 1] = c - w / 2, c + w / 2
    return out


def param_name_ok(params):
    """各パラメータ名が有効か (重複・予約語・不正な文字がないか)"""
    seen, out = {"t"}, []
    for p in params:
        n = p["name"].strip()
        ok = bool(p.get("fixed")) or bool(re.fullmatch(r"[^\W\d]\w*", n) and n not in seen and not core.param_reserved(n))
        if ok: seen.add(n)
        out.append(ok)
    return out


def view_shift_3d(L, azim, side, fwd):
    """3D: 今向いている方向を基準に、描画範囲を動かす。side: +1 右 / -1 左、fwd: +1 奥 / -1 手前 (1回で範囲の25%)。
    画面の「右」は (-sin a, cos a)、「奥」は (-cos a, -sin a) (a = 方位角)"""
    a = np.radians(float(azim)); rx, ry = -np.sin(a), np.cos(a); fx, fy = -np.cos(a), -np.sin(a)
    sx, sy = L[1] - L[0], L[3] - L[2]
    dx = (side * rx + fwd * fx) * 0.25 * sx; dy = (side * ry + fwd * fy) * 0.25 * sy
    out = list(L); out[0] += dx; out[1] += dx; out[2] += dy; out[3] += dy
    return out
