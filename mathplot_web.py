"""MathPlot Studio (Web版) - Streamlit から呼び出す描画エンジン。

mathplot_core.py（数式の解析・描画ロジック）を、PySide6(GUI)なしで使うための橋渡しです。
このファイルは streamlit に依存しないので、パソコン上で単体テストもできます。
"""
import os
os.environ.setdefault("MPS_NOGUI", "1")      # mathplot_core の GUI(PySide6) 部分を読み込まない
import io, re, threading
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
EXAMPLES = {
    "xy平面": "sin(x)",
    "複素数平面": "z^2",
    "xyz空間": "z = sin(sqrt(x^2 + y^2) - t)",
    "xy平面 + パラメータ軸t": "y = sin(x - t)",
    "複素数平面 + パラメータ軸t": "z^2 - exp(i t)",
    "極座標平面": "1 + cos(th)",
}

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


def keypages(param_names):
    """キーパッドのページ一覧。末尾に現在のパラメータ用ページを自動追加。"""
    pages = [p for p in DEFAULT_KEYS]
    pages = [dict(name=p["name"], keys=[k for k in p["keys"] if k[1] != "(|)/()"]) for p in pages]   # a/b キーは Web では不要
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
def render(state, w_in=8.0, h_in=7.0, dpi=130):
    """グラフをPNGのバイト列にして返す。戻り値: (png, 行ごとのエラー文字列のリスト, 実際に使った軸範囲)"""
    with LOCK:
        S = SPACES[state["space"]]; dim = S["dim"]; tz = S.get("tz")
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
            w, h = W_ * 0.94, H_ * 0.93
            u = max((L[1] - L[0]) / w, (L[3] - L[2]) / h); cx, cy = (L[0] + L[1]) / 2, (L[2] + L[3]) / 2
            L[:4] = [cx - u * w / 2, cx + u * w / 2, cy - u * h / 2, cy + u * h / 2]
        R = max(abs(v) for v in (L if dim == 3 and not tz else L[:4]))
        ax = fig.add_axes([0.055, 0.065, 0.94, 0.93] if dim == 2 else [0, 0, 1, 1], projection="3d" if dim == 3 else None)
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
            ax.view_init(state.get("elev", 25), state.get("azim", -60))
            core.fit3d(ax, fig, L, tuple(v / max(asp) for v in asp), 0.96)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi)
        return buf.getvalue(), errs, L
