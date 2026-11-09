#!/usr/bin/env python3
"""MathPlot Studio - 数学グラフ描画ソフト (2D/3D, 複素数平面, パラメータ軸, ベクトル, 複数ウィンドウ/タブ/パネル)

追加機能: 陰関数 / 極座標 / 全画面(⛶, Escで戻る) / JSON保存・読込 / 積分・Σ・逆三角・双曲線 / 離散格子点 / 近似値表示 / 条件(不等式)による制限 / 斜交座標
必要ライブラリ:  pip install PySide6 numpy sympy matplotlib
起動:           python mathplot_studio.py
式の書き方:     sin(x), exp(-x^2), sqrt(1-x^2), z^2+1, i*z, pi, t(パラメータ), s(曲線の媒介変数)
"""
import sys, io, re, json, os, itertools
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication_application, convert_xor)
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d

# ======================= 数式パーサ =======================
BASE_NAMES = ("x", "y", "z", "t", "s", "th", "r", "n", "m", "k")
NAMES = list(BASE_NAMES)          # 基本変数 + ユーザー追加パラメータ
SYMS = [sp.Symbol(n) for n in NAMES]
LOC = {n: s_ for n, s_ in zip(NAMES, SYMS)}
PVAL, AX, CUSTOM = {}, ["t"], set()   # パラメータの現在値 / 第3軸に使うパラメータ名 / 追加パラメータ名
LOC.update(i=sp.I, j=sp.I, e=sp.E, pi=sp.pi, ln=sp.log, abs=sp.Abs, conj=sp.conjugate,
           re=sp.re, im=sp.im, arg=sp.arg, sgn=sp.sign, arcsin=sp.asin, arccos=sp.acos, arctan=sp.atan,
           arcsinh=sp.asinh, arccosh=sp.acosh, arctanh=sp.atanh, arctan2=sp.atan2, mod=sp.Mod,
           sum=lambda f, k, a, b: sp.Sum(f, (k, a, b)), prod=lambda f, k, a, b: sp.Product(f, (k, a, b)),
           integral=lambda f, u, a, b: sp.Integral(f, (u, a, b)),
           vec=sp.Function("vec"), field=sp.Function("field"), basis=sp.Function("basis"))
TF = standard_transformations + (implicit_multiplication_application, convert_xor)
# ---- 追加: 数学関数・ベクトル・定義機構 ----
class LogB(sp.Function):
    """底つき対数 log(x, b): 表示は log_b(x) のまま保持する"""
    nargs = 2

    def _latex(self, printer):
        return r"\log_{%s}{\left(%s\right)}" % (printer._print(self.args[1]), printer._print(self.args[0]))

    def _eval_evalf(self, prec):
        return (sp.log(self.args[0]) / sp.log(self.args[1])).evalf(prec)


def _comps(x):
    """ベクトル・タプルを成分の一次元リストに正規化する。vec(Ma) のような入れ子も展開。"""
    if getattr(getattr(x, "func", None), "__name__", "") == "vec":
        a = list(x.args)
        if len(a) == 1:
            return _comps(a[0])
        if len(a) == 2 and all(isinstance(q, sp.Tuple) for q in a):
            return list(a[1])
        out = []
        for q in a: out.extend(_comps(q) if isinstance(q, (sp.Tuple, sp.MatrixBase)) else [q])
        return out
    if isinstance(x, sp.MatrixBase): return list(x)
    if isinstance(x, (sp.Tuple, tuple, list)):
        out = []
        for q in x: out.extend(_comps(q) if isinstance(q, (sp.Tuple, sp.MatrixBase)) else [q])
        return out
    return [x]


def _det(*a):
    if len(a) == 1 and isinstance(a[0], sp.MatrixBase): return a[0].det()
    return sp.Matrix([_comps(x) for x in a]).det()


def _make_vec(*args):
    """vec(a,b) / vec((a,b)) / vec(Ma) を列ベクトルへ変換"""
    return sp.ImmutableMatrix(_comps(args[0]) if len(args) == 1 else _comps(args))


LOC.update(Re=sp.re, Im=sp.im, Arg=sp.arg, Abs=sp.Abs, log=lambda x, b=None: sp.log(x) if b is None else LogB(x, b),
           log10=lambda x: LogB(x, 10), log2=lambda x: LogB(x, 2), det=_det, vec=_make_vec,
           dot=lambda u, v: sp.Matrix(_comps(u)).dot(sp.Matrix(_comps(v))),
           cross=lambda u, v: sp.ImmutableMatrix(sp.Matrix(_comps(u)).cross(sp.Matrix(_comps(v)))),
           norm=lambda u: sp.sqrt(sum(sp.Abs(q) ** 2 for q in _comps(u))))
DEFS, FNAMES, CURKEY = {}, set(), ["xy"]   # ユーザー定義 (定数・関数・ベクトル・点) / 関数名 / 現在の空間
IDENT = r"[^\W\d][\w\u20d7']*"
DEF_RE = re.compile(rf"^\s*({IDENT})\s*(?:\(\s*([^()=]*?)\s*\))?\s*=(?!=)\s*(.+?)\s*$", re.U)
PT_RE = re.compile(r"^\s*([A-ZΑ-Ω][0-9']*)\s*\((.+)\)\s*$")
MAP_RE = r"^\s*w\s*=(?!=)\s*(.+)$"


def is_reserved(name, key):
    return (name in set(NAMES) | {"i", "j", "e", "pi", "Re", "Im"} or name in LOC or (key in ("cx", "cxt") and name == "w")
            or (len(name) > 1 and hasattr(sp, name)))


def parse_def(src, key, fnames=None):
    """定義行なら (kind, name, params, rhs, cond)。kind: const / func / point。 例: α=1+√5, f(x)=x^2, a⃗=vec(1,0), A(1,0)"""
    fn = FNAMES if fnames is None else fnames
    body, cond = split_cond(src)
    m = PT_RE.match(body)
    if m and "=" not in body and m.group(1) not in fn: return ("point", m.group(1), None, "(" + m.group(2) + ")", cond)
    m = DEF_RE.match(body)
    if not m or is_reserved(m.group(1), key) or re.search(REL, m.group(3)): return None
    return ("func" if m.group(2) is not None else "const", m.group(1), m.group(2), m.group(3), cond)


def def_value(kind, params, rhs):
    if kind == "func":
        return sp.Lambda(tuple(sp.Symbol(p.strip()) for p in params.split(",") if p.strip()), compile_expr(rhs)[2])
    v = compile_expr(rhs)[2]
    # 行列式・行列積の結果はそのまま保持する。vec(...) は中身が行列でも成分へ展開する。
    if isinstance(v, sp.MatrixBase): return sp.ImmutableMatrix(v)
    if isinstance(v, sp.Tuple): return sp.ImmutableMatrix(_comps(v))
    if getattr(getattr(v, "func", None), "__name__", "") == "vec":
        return sp.ImmutableMatrix(_comps(v))
    return v


def build_defs(texts, key):
    """全行から定義を集めて DEFS を作る (定義順に依存しない)。戻り値: {行番号: エラー文字列 or None}"""
    DEFS.clear(); CURKEY[0] = key; out = {}
    FNAMES.clear(); FNAMES.update(m.group(1) for m in (DEF_RE.match(split_cond(t)[0]) for t in texts)
                                  if m and m.group(2) is not None and not is_reserved(m.group(1), key))
    todo = {i: d for i, d in ((i, parse_def(t, key)) for i, t in enumerate(texts)) if d}
    names = {d[1] for d in todo.values()}

    def mentioned_defs(expr):
        """識別子の連結 (Ma など) も既存定義名の積として認識する。"""
        found = set()
        candidates = sorted(names, key=len, reverse=True)
        for token in re.findall(IDENT, expr):
            if token in names:
                found.add(token); continue
            # token 全体を定義名の連結に分解できる場合だけ依存関係として扱う。
            reachable = {0}; pieces = {}
            for pos in range(len(token) + 1):
                if pos not in reachable: continue
                for q in candidates:
                    if token.startswith(q, pos):
                        end = pos + len(q); reachable.add(end); pieces.setdefault(end, set()).add(q)
            if len(token) in reachable:
                pos, used = len(token), set()
                while pos:
                    opts = pieces.get(pos, set())
                    q = next((z for z in sorted(opts, key=len, reverse=True) if pos-len(z) in reachable), None)
                    if q is None: break
                    used.add(q); pos -= len(q)
                if pos == 0: found.update(used)
        return found

    while todo:
        prog = False
        for i, (kind, name, params, rhs, cond) in list(todo.items()):
            ids = mentioned_defs(rhs)
            if any(q in names and q not in DEFS and q != name for q in ids): continue
            try:
                if name in ids and kind != "func": raise ValueError("自己参照です")
                DEFS[name] = def_value(kind, params, rhs); out[i] = None
            except Exception as ex: out[i] = str(ex)
            del todo[i]; prog = True
        if not prog:
            for i in todo: out[i] = "循環参照または未定義の名前があります"
            break
    return out


def latex_(e):
    """sympyのlatex。mathtextで描けない \\limits を除去し、自然対数は ln と表記"""
    if isinstance(e, sp.MatrixBase):
        if min(e.shape) == 1: return r"\left(" + r",\ ".join(latex_(q) for q in e) + r"\right)"   # ベクトルは (a, b)
        return r"\left[" + r";\ ".join(r",\ ".join(latex_(q) for q in e.row(i)) for i in range(e.rows)) + r"\right]"   # 行列は [行; 行]
    t_ = sp.latex(e).replace(r"\limits", "").replace(r"\log{", r"\ln{")
    return t_.replace(r"\bmod", r"\ \mathrm{mod}\ ").replace(r"\operatorname{", r"\mathrm{")   # mod と関数名は mathtext 対応の形で表記


def _sqrt_sub(s):
    while "√" in s:
        i = s.index("√"); j = i + 1
        while j < len(s) and s[j] == " ": j += 1
        if j < len(s) and s[j] == "(":
            d = 0
            for e in range(j, len(s)):
                d += (s[e] == "(") - (s[e] == ")")
                if d == 0: break
            s = s[:i] + "sqrt" + s[j:e + 1] + s[e + 1:]
        else:
            m = re.match(r"\d+(?:\.\d+)?|[^\W\d]\w*", s[j:])
            s = s[:i] + ("sqrt(" + m.group() + ")" + s[j + m.end():] if m else "sqrt" + s[i + 1:])
    return s

LOC.update(mat2=lambda *a: sp.ImmutableMatrix(2, 2, list(a)), mat3=lambda *a: sp.ImmutableMatrix(3, 3, list(a)),
           inv=lambda M: M.inv(), transp=lambda M: M.T)
LOC.update(eta=sp.dirichlet_eta, Li=sp.polylog, lambertw=sp.LambertW, ellipk=sp.elliptic_k, ellipe=sp.elliptic_e,
           Gamma=sp.gamma, psi=sp.digamma, hypot=lambda a, b: sp.sqrt(a ** 2 + b ** 2))
_LOC_BASE = set(LOC)

try:
    import mpmath as _mp
except Exception:   # mpmath は sympy の依存なので通常は入っている
    _mp = None


def _c(v):
    try: return complex(v)
    except Exception: return np.nan


def mpvec(f):
    """mpmath の関数を、numpy配列 (実数・複素数) を要素ごとに処理できる関数にする"""
    def g(*a):
        a = [np.asarray(q) for q in a]
        r = np.asarray(np.frompyfunc(lambda *q: _c(f(*q)), len(a), 1)(*a), dtype=complex)
        return r.real if np.all(np.abs(r.imag) <= 1e-12 * np.abs(r)) else r
    return g


# sympy の関数名 → mpmath の関数名 (名前が違うものだけ)
_MP_ALIAS = dict(elliptic_k="ellipk", elliptic_e="ellipe", LambertW="lambertw", dirichlet_eta="altzeta", Ei="ei", Si="si", Ci="ci",
                 Shi="shi", Chi="chi", polygamma="psi", lerchphi="lerchphi", expint="expint", li="li", polylog="polylog")
FALLBACK = {}      # 最低優先度の名前空間: scipy / numpy に無い関数だけここで補う
if _mp is not None:
    FALLBACK.update({k_: mpvec(v_) for k_, v_ in _mp.__dict__.items() if callable(v_) and not k_.startswith("_")})
    FALLBACK.update({sk: mpvec(getattr(_mp, mk)) for sk, mk in _MP_ALIAS.items() if hasattr(_mp, mk)})
# ---- 高速な ζ・Γ・lnΓ (numpyで一括計算。scipy が無くても複素数の格子で実用的な速さ) ----
try:
    from scipy.special import gamma as _sc_gamma, loggamma as _sc_lgamma
except Exception:
    _sc_gamma = _sc_lgamma = None
_LG = (0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
       12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
_B2J = (1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6, -3617 / 510)   # ベルヌーイ数 B2, B4, ...


def _real_if(r):
    return r.real if np.all(np.abs(r.imag) <= 1e-12 * np.abs(r)) else r   # 虚部が大きさに対して無視できるときだけ実数にする


def _lg_core(z):
    """Re z >= 0.5 の lnΓ: |z| が十分大きくなるまで漸化式で押し上げ、スターリング級数で評価 (虚部が大きくても高精度)"""
    acc = 0
    for _ in range(14):
        m_ = np.abs(z) < 14; acc = acc - np.where(m_, np.log(np.where(m_, z, 1)), 0); z = np.where(m_, z + 1, z)
    z2 = z * z
    ser = 1 / (12 * z) - 1 / (360 * z * z2) + 1 / (1260 * z * z2 * z2) - 1 / (1680 * z * z2 ** 3) + 1 / (1188 * z * z2 ** 4)
    return (z - 0.5) * np.log(z) - z + 0.5 * np.log(2 * np.pi) + ser + acc


def lgamma_np(z):
    if _sc_lgamma is not None: return _sc_lgamma(z)
    z = np.asarray(z, dtype=complex); refl = z.real < 0.5
    return _real_if(np.where(refl, np.log(np.pi) - np.log(np.sin(np.pi * np.where(refl, z, 0.5))) - _lg_core(np.where(refl, 1 - z, 1.0)),
                             _lg_core(np.where(refl, 1.0, z))))


def gamma_np(z):
    if _sc_gamma is not None: return _sc_gamma(z)
    return _real_if(np.exp(np.asarray(lgamma_np(np.asarray(z, dtype=complex)), dtype=complex)))


def zeta_np(s):
    """リーマンのゼータ関数 (オイラー・マクローリン法 + 関数等式)。実数・複素数の配列に対応"""
    s = np.asarray(s, dtype=complex); refl = s.real < 0; w = np.where(refl, 1 - s, s)
    N = int(12 + np.nanmax(np.abs(w)) / 2) if w.size else 12
    if N > 4000: return mpvec(_mp.zeta)(s)
    acc = np.zeros_like(w)
    for k in range(1, N): acc += k ** (-w)
    acc += N ** (1 - w) / (w - 1) + 0.5 * N ** (-w); term = w * N ** (-w - 1); fact = 2.0   # j=1: B2/2! * w * N^(-w-1)
    for j, b in enumerate(_B2J, 1):
        acc += b / fact * term
        term = term * (w + 2 * j - 1) * (w + 2 * j) / N ** 2; fact *= (2 * j + 1) * (2 * j + 2)
    out = np.where(refl, 2 ** s * np.pi ** (s - 1) * np.sin(np.pi * s / 2) * np.asarray(gamma_np(1 - s), dtype=complex) * acc, acc)
    return _real_if(out)


_ZETA, _GAMMA, _LGAMMA = sp.Function("_zeta"), sp.Function("_gamma"), sp.Function("_lgamma")


def _use_fast(expr):
    """ζ(s) Γ(x) lnΓ(x) を上の高速実装へ置き換える"""
    return (expr.replace(lambda q: isinstance(q, sp.zeta) and len(q.args) == 1, lambda q: _ZETA(q.args[0]))
                .replace(lambda q: isinstance(q, sp.gamma), lambda q: _GAMMA(q.args[0]))
                .replace(lambda q: isinstance(q, sp.loggamma), lambda q: _LGAMMA(q.args[0])))


FALLBACK.update(primepi=mpvec(lambda v: sp.primepi(int(v))), totient=mpvec(lambda v: sp.totient(int(v))),
                prime=mpvec(lambda v: sp.prime(int(v))), isprime=mpvec(lambda v: int(sp.isprime(int(v)))))


def set_params(names):
    """追加パラメータを式で使える変数として登録"""
    for n in CUSTOM - set(names): LOC.pop(n, None)
    CUSTOM.clear(); CUSTOM.update(n for n in names if n not in BASE_NAMES)
    NAMES[:] = list(BASE_NAMES) + sorted(CUSTOM); SYMS[:] = [sp.Symbol(n) for n in NAMES]
    for n, s_ in zip(NAMES, SYMS): LOC[n] = s_


def param_reserved(n):
    return n in BASE_NAMES or n in {"i", "j", "e", "pi"} or n in _LOC_BASE or (len(n) > 1 and hasattr(sp, n))


def _isvec(v): return isinstance(v, sp.MatrixBase) and min(v.shape) == 1


CONSTS = {}   # 定数 (パネルごとに描画前にセットされる)
_cache, _lc = {}, {}
EQ = r"(?<![<>=!])=(?!=)"
REL = r"<=|>=|≤|≥|<|>|(?<![<>=!])=(?!=)"


def norm_cond(c):
    """不等式の連鎖 (0<x<3) と and / or を sympy が読める形に変換"""
    ors = []
    for o in re.split(r"\bor\b", c):
        ands = []
        for a in re.split(r"\band\b", o):
            t = re.split(r"(<=|>=|<|>)", a)
            ands += [f"({t[i]}{t[i + 1]}{t[i + 2]})" for i in range(0, len(t) - 2, 2)]
        if ands: ors.append("(" + "&".join(ands) + ")")
    return "|".join(ors)


def prep(src):
    s_ = src.strip()
    for a, b in (("π", "pi"), ("θ", "th"), ("≤", "<="), ("≥", ">="), ("×", "*"), ("÷", "/"), ("−", "-"),
                 ("&&", " and "), ("||", " or ")): s_ = s_.replace(a, b)
    while re.search(r"\|[^|]+\|", s_): s_ = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s_)   # |z| → 絶対値
    s_ = _sqrt_sub(s_)
    s_ = re.sub(r"ζ(?=\()", "zeta", s_); s_ = re.sub(r"Γ(?=\()", "gamma", s_); s_ = re.sub(r"ψ(?=\()", "digamma", s_)   # ζ(s) Γ(x) ψ(x)
    if re.search(r"<|>|\band\b|\bor\b", s_): s_ = norm_cond(s_)
    elif re.search(EQ, s_):
        l, r_ = s_.split("=", 1); s_ = f"({l})-({r_})"
    return s_


def split_cond(src):
    """'式 {条件}' を (式, 条件) に分ける"""
    m = re.match(r"^(.*?)\s*\{(.*)\}\s*$", src.strip(), re.S)
    return (m.group(1), m.group(2)) if m else (src.strip(), "")


def _quad(expr):
    """doit()で解けない定積分をGauss-Legendre法の数値積分に置換"""
    mods = {}
    for i, I in enumerate(expr.atoms(sp.Integral)):
        (u, a, b), f = I.limits[0], I.function
        d = sp.Dummy(); g = sp.lambdify((d,) + tuple(SYMS), f.subs(u, d), "numpy")

        def q(a, b, *args, g=g):
            xs, ws = np.polynomial.legendre.leggauss(160)
            a, b = np.asarray(a, float), np.asarray(b, float); nd = max(a.ndim, b.ndim, *[np.ndim(v) for v in args])
            sh = (-1,) + (1,) * nd
            U = a + (b - a) * (xs.reshape(sh) + 1) / 2
            return (ws.reshape(sh) * g(U, *[np.asarray(v)[None] for v in args])).sum(0) * (b - a) / 2
        mods[f"_Q{i}"] = q
        expr = expr.xreplace({I: sp.Function(f"_Q{i}")(a, b, *SYMS)})
    return expr, mods


def compile_expr(src):
    """戻り値: (表示用の式, None, 評価済みの式)"""
    key = (src, tuple(sorted(CONSTS.items())), tuple(sorted(DEFS.items(), key=lambda kv: kv[0])))
    if key not in _cache:
        shown = parse_expr(prep(src), local_dict={**LOC, **DEFS}, transformations=TF)
        if isinstance(shown, tuple): shown = sp.Tuple(*shown)
        if CONSTS: shown = shown.subs({sp.Symbol(k_): v_ for k_, v_ in CONSTS.items()})
        try: val = shown.doit()
        except Exception: val = shown
        _cache[key] = (shown, None, val)
    return _cache[key]


def _nderiv(expr):
    """diff(...) で sympy が導関数の式を作れず Derivative のまま残ったもの (ζ(x) など) を、数値微分 (5点公式) に置き換える。
    定義できる範囲 (例: ζ(x) なら x>1 など) では微分でき、定義できない点は nan になる"""
    mods = {}
    for i, D in enumerate(sorted(expr.atoms(sp.Derivative), key=sp.default_sort_key)):
        steps = []
        for v_, n_ in D.variable_count:
            if v_ not in SYMS: raise ValueError("未定義の記号: " + str(v_))
            if n_ > 3: raise ValueError("数値微分は3階までです")
            steps += [SYMS.index(v_)] * int(n_)
        f = lam(D.expr)                                  # 微分される式 (さらに入れ子の微分があってもよい)

        def mk(g, idx, order):
            h0 = 1e-3 * 10 ** (0.5 * (order - 1))          # 階数が高いほど刻み幅を大きく (丸め誤差対策)

            def d(*a):
                a = [np.asarray(q) for q in a]; x = a[idx]
                h = h0 * np.maximum(1.0, np.abs(x))
                def F(xx):
                    b = list(a); b[idx] = xx; return g(*b)
                return (F(x - 2 * h) - 8 * F(x - h) + 8 * F(x + h) - F(x + 2 * h)) / (12 * h)
            return d

        g = f
        for k, idx in enumerate(steps, 1):                 # 高階微分は1階の数値微分を重ねる
            g = mk(g, idx, len(steps))
        name = f"_ND{i}"; mods[name] = lambda *a, g=g: g(*a)
        expr = expr.xreplace({D: sp.Function(name)(*SYMS)})
    return expr, mods


def lam(expr):
    key = (expr, tuple(NAMES))
    if key not in _lc:
        bad = sorted(str(q) for q in expr.free_symbols if q not in SYMS)   # 未定義の記号で巨大な記号計算に入るのを防ぐ
        if bad: raise ValueError("未定義の記号: " + ", ".join(bad))
        e2, mods = _quad(_use_fast(expr)); mods["LogB"] = lambda x, b: np.log(x) / np.log(b)
        e2, m2 = _nderiv(e2); mods.update(m2)
        mods.update(_zeta=zeta_np, _gamma=gamma_np, _lgamma=lgamma_np)
        try: f = sp.lambdify(SYMS, e2, modules=[mods, "scipy", "numpy", FALLBACK])
        except Exception: f = sp.lambdify(SYMS, e2, modules=[mods, "numpy", FALLBACK])
        f.names = tuple(NAMES); _lc[key] = f
    return _lc[key]


def evx(e, shape=None, **v):
    f = lam(e)
    if AX[0] != "t" and "t" in v: v = dict(v); v[AX[0]] = v.pop("t")     # 第3軸に選んだパラメータへ値を渡す
    with np.errstate(all="ignore"):
        r_ = np.asarray(f(*[v.get(n, PVAL.get(n, 0.0)) for n in f.names]))
    return np.broadcast_to(r_, shape) if shape is not None else r_


def ev(src, shape=None, **v): return evx(compile_expr(src)[2], shape, **v)


def num_of(text):
    """数値または定数式 (pi, 2π, -pi/2, sqrt(2), 定義済みの定数名 など) を float にする"""
    v = compile_expr(text)[2]
    if getattr(v, "free_symbols", None): raise ValueError("定数だけ使えます")
    r = complex(sp.N(v))
    if abs(r.imag) > 1e-12: raise ValueError("実数にしてください")
    return r.real


def tex_of(src, approx=True):
    """式のTeX文字列。値が確定していれば ≈近似値 を付ける"""
    shown, _, val = compile_expr(src); t_ = latex_(shown)
    if approx and getattr(val, "is_number", False) and not shown.is_Number:
        try:
            v = complex(sp.N(val, 12))
            t_ += r"\approx " + (f"{v.real:.10g}" if abs(v.imag) < 1e-12 else f"{v.real:.8g}{v.imag:+.8g}i")
        except Exception: pass
    return t_


def _tex_plain(body):
    shown = compile_expr(body)[0]; name = getattr(shown.func, "__name__", "")
    if name in ("vec", "field", "basis"): return rf"\mathrm{{{name}}}\left(" + r",\ ".join(latex_(a) for a in shown.args) + r"\right)"
    return tex_of(body)


def tex_item(src):
    """入力1行全体のTeX表示 (関係式・組・条件 {…} 対応)"""
    body, cond = split_cond(src); ops = re.findall(REL, body); d = parse_def(src, CURKEY[0])
    if d:
        kind, name, params, rhs, _ = d
        nm = (r"\vec{%s}" % name.replace("\u20d7", "")) if "\u20d7" in name else name
        tex = nm + _tex_plain(rhs) if kind == "point" else f"{nm}({params})=" + tex_of(rhs) if kind == "func" else f"{nm}=" + _tex_plain(rhs)
    elif len(ops) == 1:
        l, r_ = re.split(REL, body, maxsplit=1)
        op = {"<=": r"\leq ", ">=": r"\geq ", "≤": r"\leq ", "≥": r"\geq "}.get(ops[0], ops[0])
        tex = tex_of(l, False) + op + tex_of(r_, False)
    elif ops: tex = latex_(compile_expr(body)[0])
    else: tex = _tex_plain(body)
    if cond.strip(): tex += r"\ \left\{" + tex_of(cond, False) + r"\right\}"
    return tex


def Rl(a):
    """実数化 (虚部が残るところは欠損扱い)"""
    a = np.asarray(a)
    if np.iscomplexobj(a): return np.where(np.abs(a.imag) < 1e-9, a.real, np.nan)
    return a.astype(float)


def sc(src, **v): return float(Rl(ev(src, **v)))


def bounds(cond):
    """条件内の a<=s<=b 形式から、媒介変数・整数変数の範囲を取り出す"""
    B = {}
    for piece in re.split(r"\band\b", cond.replace("≤", "<=").replace("≥", ">=")):
        t = re.split(r"(<=|>=|<|>)", piece)
        if len(t) == 5 and t[1][0] == t[3][0] and t[2].strip() in NAMES:
            try: a, b = sc(t[0]), sc(t[4])
            except Exception: continue
            B[t[2].strip()] = (min(a, b), max(a, b))
    return B


def Mk(ctx, a, **v):
    """条件を満たさない点を欠損(nan)にする"""
    if not ctx.cond.strip(): return a
    c = np.broadcast_to(np.asarray(ev(ctx.cond, np.shape(a), **v)).astype(bool), np.shape(a))
    return np.where(c, a, np.nan)


# ======================= 描画 (入力1行 = 1つの式。種類は内容から自動判定) =======================
def grid2(ctx, n):
    g = np.linspace(-ctx.R, ctx.R, n); return np.meshgrid(g, g)


def gvars(k, X, Y, t):
    d = dict(x=X, y=Y, t=t)
    if k == "pol": d.update(r=np.hypot(X, Y), th=np.arctan2(Y, X), s=np.arctan2(Y, X))
    if k in ("cx", "cxt"): d["z"] = X + 1j * Y
    return d


def domain_rgb(e, Z, t):
    W_ = np.asarray(evx(e, Z.shape, z=Z, x=Z.real, y=Z.imag, t=t), dtype=complex)
    h = np.nan_to_num((np.angle(W_) / (2 * np.pi)) % 1)
    v = np.nan_to_num(0.55 + 0.45 * (np.log2(np.abs(W_) + 1e-300) % 1))
    return mcolors.hsv_to_rgb(np.dstack([h, np.ones_like(h), v]))


def dom2(ax, e, ctx):
    X, Y = grid2(ctx, 600); Z = X + 1j * Y
    a = np.where(np.isnan(Mk(ctx, np.ones(X.shape), x=X, y=Y, z=Z, t=ctx.t)), 0, 0.85)
    ax.imshow(np.dstack([domain_rgb(e, Z, ctx.t), a]), extent=[-ctx.R, ctx.R, -ctx.R, ctx.R], origin="lower")


def dom3(ax, e, ctx):
    X, Y = grid2(ctx, 140); Z = X + 1j * Y
    rgb = np.dstack([domain_rgb(e, Z, ctx.t), np.full(X.shape, 0.9)])
    ax.plot_surface(X, Y, np.full_like(X, ctx.t), facecolors=rgb, shade=False, rstride=1, cstride=1)


def surf3(ax, e, ctx, c):   # z = f(x,y,t)
    X, Y = grid2(ctx, 80); v = dict(x=X, y=Y, t=ctx.t)
    Z = np.clip(Rl(evx(e, X.shape, **v)), -3 * ctx.R, 3 * ctx.R); Z = Mk(ctx, Z, z=Z, **v)
    ax.plot_surface(X, Y, Z, color=c, alpha=0.8, linewidth=0, antialiased=True)


def surf_t(ax, e, ctx, c):  # y = f(x,t) を (x, y, t) 空間の曲面に
    X, T = np.meshgrid(np.linspace(-ctx.R, ctx.R, 120), np.linspace(*ctx.tr, 80)); v = dict(x=X, t=T)
    Y = np.clip(Rl(evx(e, X.shape, **v)), -3 * ctx.R, 3 * ctx.R); Y = Mk(ctx, Y, y=Y, **v)
    ax.plot_surface(X, Y, T, color=c, alpha=0.8, linewidth=0)


def imp3(ax, e, ctx, c):
    g = np.linspace(-ctx.R, ctx.R, 56); X, Y, Z = np.meshgrid(g, g, g, indexing="ij"); v = dict(x=X, y=Y, z=Z, t=ctx.t)
    V = Mk(ctx, Rl(evx(e, X.shape, **v)), **v)
    try:
        from skimage.measure import marching_cubes
        vt, f, _, _ = marching_cubes(np.nan_to_num(V, nan=1e9), 0, spacing=(g[1] - g[0],) * 3); vt -= ctx.R
        ax.plot_trisurf(vt[:, 0], vt[:, 1], vt[:, 2], triangles=f, color=c, alpha=0.8, linewidth=0)
    except ImportError:   # scikit-image が無ければ符号変化点の点群で近似
        S_ = np.sign(V); i = np.argwhere(S_[:-1] * S_[1:] <= 0)
        ax.scatter(g[i[:, 0]], g[i[:, 1]], g[i[:, 2]], s=3, color=c)


def implicit(ax, e, k, ctx, c):   # e = 0
    if k == "xyz": return imp3(ax, e, ctx, c)
    if k in ("xyt", "cxt"): raise ValueError("この空間では陰関数・不等式は使えません")
    X, Y = grid2(ctx, 700); v = gvars(k, X, Y, ctx.t)
    V = Mk(ctx, Rl(evx(e, X.shape, **v)), **v)
    ax.contour(X, Y, np.ma.masked_invalid(V), [0], colors=[c], linewidths=ctx.lw)


def region(ax, body, k, ctx, c):   # 不等式 → 領域を塗る
    if k == "xyz":
        g = np.linspace(-ctx.R, ctx.R, 28); X, Y, Z = np.meshgrid(g, g, g); v = dict(x=X, y=Y, z=Z, t=ctx.t)
        m = np.broadcast_to(np.asarray(ev(body, X.shape, **v)).astype(bool), X.shape) & ~np.isnan(Mk(ctx, np.zeros(X.shape), **v))
        return ax.scatter(X[m], Y[m], Z[m], s=6 * ctx.lw, color=c, alpha=0.25)
    if k in ("xyt", "cxt"): raise ValueError("この空間では陰関数・不等式は使えません")
    X, Y = grid2(ctx, 500); v = gvars(k, X, Y, ctx.t)
    m = np.broadcast_to(np.asarray(ev(body, X.shape, **v)).astype(bool), X.shape).astype(float)
    m = np.where(np.isnan(Mk(ctx, np.ones(X.shape), **v)), 0.0, m)
    ax.contourf(X, Y, m, [0.5, 1.5], colors=[c], alpha=0.3); ax.contour(X, Y, m, [0.5], colors=[c], linewidths=ctx.lw)


def draw_rel(ax, body, k, ctx, c, B):
    ops = re.findall(REL, body); R, t = ctx.R, ctx.t
    if ops != ["="]: return region(ax, body, k, ctx, c)
    l, r_ = [p.strip() for p in re.split(REL, body, maxsplit=1)]
    el, er = compile_expr(l)[2], compile_expr(r_)[2]; fr = {str(q) for q in er.free_symbols}
    if k in ("xy", "cx") and l in ("y", "x") and l not in fr:      # y=f(x) / x=f(y)
        u = np.linspace(-R, R, 3000); w = Rl(evx(er, u.shape, **{"x" if l == "y" else "y": u, "t": t}))
        X_, Y_ = (u, w) if l == "y" else (w, u); w = Mk(ctx, w, x=X_, y=Y_, t=t)
        return ax.plot(*((u, w) if l == "y" else (w, u)), color=c)
    if k == "pol" and l == "r" and "r" not in fr and not fr & {"x", "y"}:   # r=f(θ)
        a, b = B.get("th", B.get("s", (0, 2 * np.pi))); th = np.linspace(a, b, 4000)
        rr = Mk(ctx, Rl(evx(er, th.shape, th=th, s=th, t=t)), th=th, s=th, t=t)
        return ax.plot(rr * np.cos(th), rr * np.sin(th), color=c)
    if k == "xyz" and l == "z" and "z" not in fr: return surf3(ax, er, ctx, c)
    if k == "xyt" and l == "y" and "y" not in fr: return surf_t(ax, er, ctx, c)
    implicit(ax, el - er, k, ctx, c)


def sample_param(items, fs, k, ctx, c, B):
    """組 (x(s),y(s)) など: s → 曲線 / n,m,k → 格子点 / 定数のみ → 点 / t軸の空間 → tを高さとする曲線"""
    t, tax, line = ctx.t, k in ("xyt", "cxt"), True
    if tax:
        tt = np.linspace(*ctx.tr, 2000); v = dict(t=tt, s=tt); shp = tt.shape
    elif fs & {"n", "m", "k"}:
        nm = [q for q in ("n", "m", "k") if q in fs]; rg = [B.get(q, (-10, 10)) for q in nm]
        gr = np.meshgrid(*[np.arange(int(np.ceil(a)), int(np.floor(b)) + 1) for a, b in rg], indexing="ij")
        v = dict(zip(nm, gr)); v["t"] = t; shp = gr[0].shape; line = False
    elif fs & {"s", "th"}:
        a, b = B.get("s", B.get("th", (-10, 10))); s = np.linspace(a, b, 4000); v = dict(s=s, th=s, t=t); shp = s.shape
    else:
        v = dict(t=t); shp = (1,); line = False
    cplx = k in ("cx", "cxt") and len(items) == 1
    if cplx:
        w = np.asarray(evx(items[0], shp, **v), dtype=complex); P = [w.real, w.imag]
    else:
        P = [Rl(evx(q, shp, **v)) for q in items]
    ex = {}
    if k == "pol": rr, th_ = P; P = [rr * np.cos(th_), rr * np.sin(th_)]; ex = dict(r=rr)
    m = dict(v); m.update(x=P[0], y=P[1], **ex)
    if len(P) == 3: m["z"] = P[2]
    if cplx: m["z"] = w
    msk = Mk(ctx, np.zeros(shp), **m)
    return [p + msk for p in P] + ([tt] if tax else []), line, shp, tax


def draw_param(ax, items, fs, k, ctx, c, B):
    co, line, shp, tax = sample_param(items, fs, k, ctx, c, B)
    if line: ax.plot(*co, color=c)
    else: ax.scatter(*co, s=(4 if int(np.prod(shp)) > 1 else 12) * ctx.lw ** 2, color=c)


def draw_point(ax, name, rhs, k, ctx, c):
    """名前つきの点 A(1,0) / A=(1,0): 点を打ち、名前を添える"""
    val = compile_expr(rhs)[2]; items = list(val.args) if isinstance(val, sp.Tuple) else list(val) if isinstance(val, sp.MatrixBase) else [val]
    ctx.cond = ""; draw_param(ax, items, {str(q) for it in items for q in it.free_symbols}, k, ctx, c, {})
    q = nums(items, ctx); xy = [q[0].real, q[0].imag] if k in ("cx", "cxt") and len(q) == 1 else [a.real for a in q]
    if k == "pol": xy = [xy[0] * np.cos(xy[1]), xy[0] * np.sin(xy[1])]
    if k in ("xyt", "cxt"): xy = xy[:2] + [ctx.t]
    ax.text(*xy, "  " + name.replace("\u20d7", ""), fontsize=11, fontweight="bold", color=c)


def draw_def(ax, d, cond, S, ctx, c):
    kind, name, params, rhs, _ = d; k = S["key"]; ctx.cond = ""
    cs = f" {{{cond}}}" if cond else ""
    if kind == "func":   # f(x)=… は定義と同時にそのグラフも描く
        vs = {"xy": ["x"], "xyt": ["x"], "cx": ["z"], "cxt": ["z"], "pol": ["th"], "xyz": ["x", "y"]}[k]
        if len([p for p in (params or "").split(",") if p.strip()]) == len(vs): draw_item(ax, f"{name}({','.join(vs)})" + cs, S, ctx, c)
        return
    if kind == "point":
        draw_point(ax, name, rhs, k, ctx, c)
        return
    value = DEFS.get(name)
    if _isvec(value):
        # ベクトル定義は点ではなく、原点を始点とする矢印として描画する。
        comps = list(value)
        args = [comps[0] + sp.I * comps[1]] if k in ("cx", "cxt") and len(comps) == 2 else comps
        draw_vec(ax, args, k, ctx, c)
        q = nums(args, ctx)
        if k in ("cx", "cxt"):
            z = q[0] if len(q) == 1 else complex(q[0].real, q[1].real)
            pos = (z.real, z.imag) if k == "cx" else (z.real, z.imag, ctx.t)
        else:
            pos = tuple(a.real for a in q)
            if k == "xyt": pos = pos[:2] + (ctx.t,)
        ax.text(*pos, "  " + name.replace("\u20d7", ""), fontsize=11, fontweight="bold", color=c)


def is_zset(body):
    """複素数平面で w=f(z) の写像を受ける z 平面上の集合か (関係式・組・曲線・点)"""
    if re.search(REL, body): return True
    val = compile_expr(body)[2]
    if getattr(val.func, "__name__", "") in KIND or isinstance(val, sp.MatrixBase): return False
    fs = {str(q) for q in val.free_symbols}
    return isinstance(val, sp.Tuple) or bool(fs & {"s", "n", "m"}) or not fs & {"z", "x", "y"}


def zset_geom(body, ctx, B):
    ops = re.findall(REL, body)
    if ops:
        eq = ops == ["="]; X, Y = grid2(ctx, 700 if eq else 800); v = gvars("cx", X, Y, ctx.t)
        if eq:
            import contourpy
            l, r_ = [p.strip() for p in re.split(REL, body, maxsplit=1)]
            V = Mk(ctx, Rl(evx(compile_expr(l)[2] - compile_expr(r_)[2], X.shape, **v)), **v)
            return [("line", ln[:, 0] + 1j * ln[:, 1]) for ln in contourpy.contour_generator(X, Y, np.ma.masked_invalid(V)).lines(0.0)]
        m = np.broadcast_to(np.asarray(ev(body, X.shape, **v)).astype(bool), X.shape) & ~np.isnan(Mk(ctx, np.zeros(X.shape), **v))
        return [("mask", X, Y, m)]
    val = compile_expr(body)[2]; items = list(val.args) if isinstance(val, sp.Tuple) else [val]
    co, line, _, _ = sample_param(items, {str(q) for it in items for q in it.free_symbols}, "cx", ctx, None, B)
    return [("line" if line else "pts", (co[0] + 1j * co[1]).ravel())]


def draw_zset(ax, body, ctx, c, B):
    """w=f(z) と z平面の条件 (|z|=1, Re(z)>0 など) から、像 w の存在範囲を図示 (元の集合は薄く表示)"""
    R = ctx.R
    for f in ctx.maps:
        for g in zset_geom(body, ctx, B):
            if g[0] == "mask":
                _, X, Y, m = g; Z = (X + 1j * Y)[m]; ax.contourf(X, Y, m.astype(float), [0.5, 1.5], colors=[c], alpha=0.07)
            else:
                Z = g[1]
                if g[0] == "line": ax.plot(Z.real, Z.imag, color=c, alpha=0.3, lw=ctx.lw * 0.6, ls="--")
                else: ax.scatter(Z.real, Z.imag, s=2 * ctx.lw ** 2, color=c, alpha=0.3)
            if Z.size == 0: continue
            w = np.asarray(evx(f, Z.shape, z=Z, x=Z.real, y=Z.imag, t=ctx.t), dtype=complex)
            w = np.where(np.abs(w) < 30 * R, w, np.nan)
            if g[0] == "line": ax.plot(w.real, w.imag, color=c)
            elif g[0] == "pts": ax.scatter(w.real, w.imag, s=4 * ctx.lw ** 2, color=c)
            else:   # 領域の像: w平面を格子に区切って占有マスクを塗る
                nb = 500; ok = np.isfinite(w)
                H, _, _ = np.histogram2d(w.imag[ok], w.real[ok], bins=nb, range=[[-R, R], [-R, R]]); occ = H > 0
                for _ in range(2): occ = occ | np.roll(occ, 1, 0) | np.roll(occ, -1, 0) | np.roll(occ, 1, 1) | np.roll(occ, -1, 1)
                occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = False
                gx = np.linspace(-R, R, nb); XX, YY = np.meshgrid(gx, gx); oc = occ.astype(float)
                ax.contourf(XX, YY, oc, [0.5, 1.5], colors=[c], alpha=0.35); ax.contour(XX, YY, oc, [0.5], colors=[c], linewidths=ctx.lw)


def draw_val(ax, items, istuple, k, ctx, c, B):
    fs = {str(q) for it in items for q in it.free_symbols}
    if istuple:
        need = {"xy": 2, "cx": 2, "pol": 2, "xyz": 3, "xyt": 2, "cxt": 2}[k]
        if len(items) != need: raise ValueError(f"この空間の組の成分は {need} 個です")
        return draw_param(ax, items, fs, k, ctx, c, B)
    e = items[0]
    if not fs and k in ("xy", "xyz", "xyt", "pol"): return    # 定数は描かない (値は式の横に表示)
    if k == "xy":
        if "y" in fs: return implicit(ax, e, k, ctx, c)           # F(x,y)=0
        x = np.linspace(-ctx.R, ctx.R, 3000); y = Rl(evx(e, x.shape, x=x, t=ctx.t))   # y=f(x)
        return ax.plot(x, Mk(ctx, y, x=x, y=y, t=ctx.t), color=c)
    if k == "xyt": return surf_t(ax, e, ctx, c)
    if k == "xyz": return implicit(ax, e, k, ctx, c) if "z" in fs else surf3(ax, e, ctx, c)
    if k == "pol":
        if fs & {"x", "y", "r"}: return implicit(ax, e, k, ctx, c)
        a, b = B.get("th", B.get("s", (0, 2 * np.pi))); th = np.linspace(a, b, 4000)   # r=f(θ)
        rr = Mk(ctx, Rl(evx(e, th.shape, th=th, s=th, t=ctx.t)), th=th, s=th, t=ctx.t)
        return ax.plot(rr * np.cos(th), rr * np.sin(th), color=c)
    if k == "cx":
        return dom2(ax, e, ctx) if (fs & {"z", "x", "y"}) and not fs & {"s", "n", "m"} else draw_param(ax, items, fs, k, ctx, c, B)
    if k == "cxt":
        return dom3(ax, e, ctx) if fs & {"z", "x", "y"} else draw_param(ax, items, fs, k, ctx, c, B)


def nums(args, ctx):
    out = []
    for a in args: out += list(a.args) if isinstance(a, sp.Tuple) else [a]
    return [complex(evx(q, None, t=ctx.t)) for q in out]


def _q2(ax, x0, y0, dx, dy, c, lw):
    ax.quiver(x0, y0, dx, dy, angles="xy", scale_units="xy", scale=1, width=0.0025 * lw, color=c)


def draw_vec(ax, args, k, ctx, c):
    v = nums(args, ctx); lw, t = ctx.lw, ctx.t; q = [a.real for a in v]
    if k in ("cx", "cxt"):
        z0, w = (0j, v[0]) if len(v) == 1 else (v[0], v[1]); h = v[2].real if len(v) > 2 else t
        if k == "cx": return _q2(ax, z0.real, z0.imag, w.real, w.imag, c, lw)
        return ax.quiver(z0.real, z0.imag, h, w.real, w.imag, 0, arrow_length_ratio=0.2, linewidth=lw, color=c)
    if k in ("xy", "pol"):
        q = [0, 0] + q if len(q) == 2 else q
        if len(q) != 4: raise ValueError("vec は 成分2個 または (始点x,始点y,成分x,成分y)")
        return _q2(ax, *q, c, lw)
    if k == "xyz":
        q = [0, 0, 0] + q if len(q) == 3 else q
        if len(q) != 6: raise ValueError("vec は 成分3個 または 始点3+成分3")
        return ax.quiver(*q, arrow_length_ratio=0.2, linewidth=lw, color=c)
    if len(q) == 2: x0, y0, dx, dy, h = 0, 0, *q, t
    elif len(q) == 4: x0, y0, dx, dy = q; h = t
    elif len(q) == 5: x0, y0, h, dx, dy = q
    else: raise ValueError("vec は 2, 4, 5 個の成分")
    ax.quiver(x0, y0, h, dx, dy, 0, arrow_length_ratio=0.2, linewidth=lw, color=c)


def draw_field(ax, args, k, ctx, c):
    R, t, lw = ctx.R, ctx.t, ctx.lw
    if k in ("xy", "pol", "cx"):
        X, Y = grid2(ctx, 17); v = gvars(k, X, Y, t)
        if k == "cx": w = np.asarray(evx(args[0], X.shape, **v), dtype=complex); U, V = w.real, w.imag
        else: U, V = [Rl(evx(a, X.shape, **v)) for a in args[:2]]
        ax.quiver(X, Y, Mk(ctx, U, **v), V, width=0.0025 * lw, color=c)
    elif k == "xyz":
        g = np.linspace(-R, R, 6); X, Y, Z = np.meshgrid(g, g, g); v = dict(x=X, y=Y, z=Z, t=t)
        U, V, W_ = [Rl(evx(a, X.shape, **v)) for a in args[:3]]
        ax.quiver(X, Y, Z, U, V, W_, length=R / 4, normalize=True, linewidth=lw, color=c)
    else: raise ValueError("この空間ではベクトル場は使えません")


def _oblique(ax, e1, e2, N, c, lw=2.0):
    """基底 e1,e2 による斜交座標格子 (a e1 + b e2, a,b は整数) と基底ベクトル"""
    e1, e2 = np.array(e1, float), np.array(e2, float)
    for a in range(-N, N + 1):
        for p, q in ((e1, e2), (e2, e1)):
            ax.plot(*zip(a * p - N * q, a * p + N * q), color=c, lw=lw * (0.45 if a else 1), alpha=0.5)
    for e in (e1, e2):
        if len(e) == 2: ax.quiver(0, 0, *e, angles="xy", scale_units="xy", scale=1, color=c, width=0.003 * lw)
        else: ax.quiver(0, 0, 0, *e, arrow_length_ratio=0.15, linewidth=lw, color=c)


def draw_basis(ax, args, k, ctx, c):
    if k in ("xyt", "cxt"): raise ValueError("斜交座標はこの空間では使えません")
    N = int(nums([args[2]], ctx)[0].real) if len(args) > 2 else 5; es = []
    for a in args[:2]:
        v = nums([a], ctx); es.append([v[0].real, v[0].imag] if k == "cx" else [q.real for q in v])
    _oblique(ax, es[0], es[1], N, c, ctx.lw)


KIND = {"vec": draw_vec, "field": draw_field, "basis": draw_basis}


def draw_item(ax, src, S, ctx, c):
    body, cond = split_cond(src)
    if not body.strip(): return
    k = S["key"]; d = parse_def(src, k)
    if d: return draw_def(ax, d, cond, S, ctx, c)
    if k == "cx" and re.match(MAP_RE, body): return          # w=f(z) 自体は描かず、他の式の像に使う
    ctx.cond = cond; B = bounds(cond)
    if k == "cx" and getattr(ctx, "maps", None) and is_zset(body): return draw_zset(ax, body, ctx, c, B)
    if re.search(REL, body): return draw_rel(ax, body, k, ctx, c, B)
    val = compile_expr(body)[2]
    if isinstance(val, sp.MatrixBase):    # ベクトル値の式 (a⃗, a+b など) は原点からの矢印。行列は描かない
        if not _isvec(val): return
        cm = list(val); return draw_vec(ax, [cm[0] + sp.I * cm[1]] if k in ("cx", "cxt") else cm, k, ctx, c)
    name = getattr(val.func, "__name__", "")
    if name == "vec": return draw_vec(ax, _comps(val), k, ctx, c)
    if name in KIND: return KIND[name](ax, val.args, k, ctx, c)
    draw_val(ax, list(val.args) if isinstance(val, sp.Tuple) else [val], isinstance(val, sp.Tuple), k, ctx, c, B)


# ======================= 軸の描画 (矢印・目盛を軸上に表示) / 3Dの全面表示 =======================
def nice_ticks(lo, hi, n=10):
    return [float(v) for v in MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi) if lo - 1e-9 <= v <= hi + 1e-9]


def axes2d(ax, L, lb, ny=8):
    """2D: 正方向へ伸びる長い矢印の軸。目盛は軸の上に、軸名は矢印の先端に表示"""
    x0, x1, y0, y1 = L[:4]; sx, sy = x1 - x0, y1 - y0
    ox, oy = min(max(0, x0), x1), min(max(0, y0), y1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    xt, yt = nice_ticks(x0, x1, 10), nice_ticks(y0, y1, ny)
    for v in xt: ax.axvline(v, color="#dcdcdc", lw=0.6, zorder=0)
    for v in yt: ax.axhline(v, color="#dcdcdc", lw=0.6, zorder=0)
    ax.plot(xt, [oy] * len(xt), "|", color="k", ms=7, mew=1.2, clip_on=False, zorder=4)
    ax.plot([ox] * len(yt), yt, "_", color="k", ms=7, mew=1.2, clip_on=False, zorder=4)
    kw = dict(annotation_clip=False, textcoords="offset points", fontsize=9, color="#222", zorder=5)
    low, lef = oy > y0 + 0.03 * sy, ox > x0 + 0.03 * sx        # 軸が端でなければ目盛は下/左、端なら内側
    for v in xt:
        if abs(v) > 1e-9 * sx and x0 + 0.03 * sx < v < x1 - 0.03 * sx: ax.annotate(f"{v:g}", (v, oy), xytext=(0, -7 if low else 7), ha="center", va="top" if low else "bottom", **kw)
    for v in yt:
        if abs(v) > 1e-9 * sy and y0 + 0.03 * sy < v < y1 - 0.03 * sy: ax.annotate(f"{v:g}", (ox, v), xytext=(-7 if lef else 7, 0), ha="right" if lef else "left", va="center", **kw)
    if x0 < 0 < x1 and y0 < 0 < y1: ax.annotate("0", (0, 0), xytext=(-6, -6), ha="right", va="top", **kw)
    ap = dict(arrowstyle="-|>", color="k", lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=16)
    ax.annotate("", xy=(x1, oy), xytext=(x0, oy), arrowprops=ap, annotation_clip=False, zorder=6)
    ax.annotate("", xy=(ox, y1), xytext=(ox, y0), arrowprops=ap, annotation_clip=False, zorder=6)
    fw = dict(fontweight="bold", fontsize=13, clip_on=False, zorder=7)
    ax.text(x1 - 0.012 * sx, oy + 0.015 * sy, lb[0], ha="right", va="bottom", **fw)
    ax.text(ox + 0.012 * sx, y1 - 0.012 * sy, lb[1], ha="left", va="top", **fw)


def axes3d(ax, L, lb):
    """3D: 箱・目盛を消し、原点を通る正方向の矢印軸 (目盛つき) と床グリッドだけを描く"""
    ax.set_axis_off()
    o = [min(max(0, L[2 * i]), L[2 * i + 1]) for i in range(3)]; sp3 = [L[2 * i + 1] - L[2 * i] for i in range(3)]
    tk = [nice_ticks(L[2 * i], L[2 * i + 1], 8) for i in range(3)]
    for v in tk[0]: ax.plot([v, v], [L[2], L[3]], [o[2]] * 2, color="#cfcfcf", lw=0.5)
    for v in tk[1]: ax.plot([L[0], L[1]], [v, v], [o[2]] * 2, color="#cfcfcf", lw=0.5)
    for i in range(3):
        p0, d = list(o), [0, 0, 0]; p0[i] = L[2 * i]; d[i] = sp3[i]
        ax.quiver(*p0, *d, arrow_length_ratio=0.035, color="k", linewidth=2)
        tip = list(o); tip[i] = L[2 * i + 1] + 0.04 * sp3[i]
        ax.text(*tip, lb[i], fontweight="bold", fontsize=13, ha="center", va="center")
        j = (1, 0, 0)[i]
        for v in tk[i]:
            if abs(v) < 1e-9 * sp3[i]: continue
            q = list(o); q[i] = v; a, b = list(q), list(q); a[j] -= 0.012 * sp3[j]; b[j] += 0.012 * sp3[j]
            ax.plot(*zip(a, b), color="k", lw=1)
            q[j] -= 0.035 * sp3[j]; ax.text(*q, f"{v:g}", fontsize=8, ha="center", va="center", color="#222")


def fit3d(ax, fig, L, asp, target=0.96):
    """3D軸の『描画領域を正方形に縮める』処理を無効化し、立方体全体が表示欄いっぱいに収まるよう拡大・中心合わせ"""
    W_, H_ = fig.bbox.width, fig.bbox.height
    if W_ < 50 or H_ < 50: return
    d = getattr(ax, "_dist", 10); k = W_ / H_; ax.apply_aspect = lambda position=None: None
    wl, hl = 1.85 / d * k, 1.85 / d; off = [0.0, 0.0]

    def setv():
        ax.viewLim.intervalx = (-0.95 / d * k + off[0], 0.9 / d * k + off[0]); ax.viewLim.intervaly = (-0.95 / d + off[1], 0.9 / d + off[1])

    def corners():
        M = ax.get_proj(); p = []
        o = [min(max(0, L[2 * i]), L[2 * i + 1]) for i in range(3)]
        pts = [tuple(L[2 * i + e] if j == i else o[j] for j in range(3)) for i in range(3) for e in (0, 1)]   # 軸の両端 (見える骨組み)
        pts += [(x, y, o[2]) for x in (L[0], L[1]) for y in (L[2], L[3])]                                     # 床グリッドの四隅
        for x, y, z in pts:
            x2, y2, _ = proj3d.proj_transform(x, y, z, M); p.append(ax.transData.transform((x2, y2)))
        p = np.array(p); return p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()
    setv(); zoom = 1.0
    for _ in range(5):
        ax.set_box_aspect(asp, zoom=zoom); x0, x1, y0, y1 = corners()
        zoom *= min(W_ * target / (x1 - x0), H_ * target / (y1 - y0))
        ax.set_box_aspect(asp, zoom=zoom); x0, x1, y0, y1 = corners()
        off[0] += ((x0 + x1) / 2 - W_ / 2) * wl / W_; off[1] += ((y0 + y1) / 2 - H_ / 2) * hl / H_; setv()


# ======================= 空間の定義 =======================
SPACES = {
    "xy平面": dict(key="xy", dim=2, labels=("x", "y"), equal=True,
                 defaults=["y = sin(x - t)", "x^2 + y^2 = 4 {y>0}", "vec((0,0),(2,1))"]),
    "複素数平面": dict(key="cx", dim=2, labels=("Re", "Im"), equal=True,
                  defaults=["(z - 1)/(z^2 + z + 1)", "2exp(i s) {0<=s<=2pi}"]),
    "xyz空間": dict(key="xyz", dim=3, labels=("x", "y", "z"), equal=True,
                  defaults=["z = sin(sqrt(x^2 + y^2) - t)", "(3cos(s), 3sin(s), s/2) {0<=s<=4pi}"]),
    "xy平面 + パラメータ軸t": dict(key="xyt", dim=3, labels=("x", "y", "t"), equal=True, tz=True,
                            defaults=["y = sin(x - t)", "(3cos(t), 3sin(t))"]),
    "複素数平面 + パラメータ軸t": dict(key="cxt", dim=3, labels=("Re", "Im", "t"), equal=True, tz=True,
                              defaults=["(1 + t/3) exp(i t)", "z^2 - exp(i t)"]),
    "極座標平面": dict(key="pol", dim=2, labels=("x", "y"), equal=True, polar=True,
                  defaults=["1 + cos(th)", "(n, m pi/6) {0<=n<=4 and 0<=m<=11}"]),
}
for _S in SPACES.values(): _S["defaults"] = []   # 起動時は何も表示しない

if __name__ != "__main__" and "MPS_NOGUI" in __import__("os").environ:
    pass  # テスト用: GUI部分を読み込まない
else:
    # ======================= GUI =======================
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib import mathtext
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT as Toolbar
    from PySide6 import QtCore, QtGui, QtWidgets as W
    from PySide6.QtCore import Qt

    PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    _ci = [0]
    WHITE = "background:white; color:black; padding:2px;"

    @lru_cache(maxsize=512)
    def tex_pixmap(tex, dpi=105):
        """TeX文字列をmathtextで描画してQPixmapにする (失敗時はNone)"""
        try:
            buf = io.BytesIO()
            mathtext.math_to_image(f"${tex}$", buf, dpi=dpi, format="png")
            pm = QtGui.QPixmap(); pm.loadFromData(buf.getvalue()); return pm
        except Exception:
            return None

    def tex_label(tex, dpi=105):
        l = W.QLabel(); l.setStyleSheet(WHITE); pm = tex_pixmap(tex, dpi)
        l.setPixmap(pm) if pm else l.setText(tex)
        return l

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
    # --- 設定ファイルの置き場所: プログラム(または exe)と同じフォルダ。書き込めない場合のみ ~/.mathplot_studio/ ---
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    BUNDLE_DIR = getattr(sys, "_MEIPASS", APP_DIR)        # PyInstaller が同梱ファイルを展開する場所 (読み取り専用の既定値)
    HOME_DIR = os.path.join(os.path.expanduser("~"), ".mathplot_studio")

    def _writable_dir():
        for d in (APP_DIR, HOME_DIR):
            try:
                os.makedirs(d, exist_ok=True); t_ = os.path.join(d, ".write_test")
                with open(t_, "w"): pass
                os.remove(t_); return d
            except OSError: continue
        return APP_DIR

    CFG_DIR = _writable_dir()
    def cfg_path(name): return os.path.join(CFG_DIR, name)
    def first_existing(*paths): return next((q for q in paths if q and os.path.exists(q)), None)

    KEYFILE = cfg_path("mathplot_studio_keys.json"); KEYPAGES = []
    LEGACY_KEYFILE = os.path.join(os.path.expanduser("~"), ".mathplot_studio_keys.json")   # 旧バージョンの保存先
    SETFILE = cfg_path("mathplot_studio_settings.json")
    SETTINGS = {"detach": True, "window_size": [1200, 800], "default_space": "xy平面", "last_dir": ""}

    def save_settings():
        try:
            with open(SETFILE, "w", encoding="utf-8") as f: json.dump(SETTINGS, f, ensure_ascii=False, indent=1)
        except OSError: pass

    def load_settings():
        try:
            with open(first_existing(SETFILE, os.path.join(BUNDLE_DIR, "mathplot_studio_settings.json")), encoding="utf-8") as f:
                SETTINGS.update({k_: v_ for k_, v_ in json.load(f).items() if k_ in SETTINGS})
        except Exception: pass
        if not os.path.exists(SETFILE): save_settings()     # 初回起動時に設定ファイルを作成

    def pick_save(parent, title, name):
        p_, _ = W.QFileDialog.getSaveFileName(parent, title, os.path.join(SETTINGS.get("last_dir") or "", name), "JSON (*.json)")
        if p_: SETTINGS["last_dir"] = os.path.dirname(p_); save_settings()
        return p_

    def pick_open(parent, title):
        p_, _ = W.QFileDialog.getOpenFileName(parent, title, SETTINGS.get("last_dir") or "", "JSON (*.json)")
        if p_: SETTINGS["last_dir"] = os.path.dirname(p_); save_settings()
        return p_

    load_settings()

    KEYS_VER = 3; NEWPAGES = {"行列・ベクトル", "特殊関数"}

    def load_keys():
        try:
            with open(first_existing(KEYFILE, LEGACY_KEYFILE, os.path.join(BUNDLE_DIR, "mathplot_studio_keys.json")), encoding="utf-8") as f: raw = json.load(f)
            data, ver = (raw["pages"], raw.get("v", 1)) if isinstance(raw, dict) else (raw, 1)
            assert data and all("name" in p and "keys" in p for p in data)
            if ver < KEYS_VER:   # 新しい既定ページを追加して保存し直す
                have = {p["name"] for p in data}; data = data + [p for p in DEFAULT_KEYS if p["name"] in NEWPAGES and p["name"] not in have]
        except Exception: data = DEFAULT_KEYS
        KEYPAGES[:] = json.loads(json.dumps(data)); save_keys()

    def save_keys():
        try:
            with open(KEYFILE, "w", encoding="utf-8") as f: json.dump({"v": KEYS_VER, "pages": KEYPAGES}, f, ensure_ascii=False, indent=1)
        except OSError: pass

    class _Bus(QtCore.QObject): changed = QtCore.Signal()
    BUS = _Bus(); load_keys()

    class ObjRow(W.QFrame):
        """Desmos風の式カード: 1行の入力 + TeXプレビュー。入力欄をクリックするとカードが選択状態になる"""
        changed = QtCore.Signal(); removed = QtCore.Signal(object); enter = QtCore.Signal(object)

        def __init__(self, text="", pre=lambda: None, color=None, on=True, lw=2.0):
            super().__init__(); self.setObjectName("row"); self.pre = pre
            self.color = color or PALETTE[_ci[0] % len(PALETTE)]; _ci[0] += 1
            v = W.QVBoxLayout(self); v.setContentsMargins(6, 4, 6, 4); h = W.QHBoxLayout()
            self.chk = W.QPushButton(); self.chk.setCheckable(True); self.chk.setChecked(on); self.chk.setFixedSize(26, 26)
            self.chk.setToolTip("表示 / 非表示"); self.chk.toggled.connect(lambda _: (self.style_btn(), self.changed.emit()))
            self.edit = W.QLineEdit(text); f = self.edit.font(); f.setPointSize(f.pointSize() + 3); self.edit.setFont(f)
            self.edit.setPlaceholderText("式を入力"); self.edit.textChanged.connect(self.on_edit)
            self.edit.returnPressed.connect(lambda: self.enter.emit(self))
            rm = W.QPushButton("×"); rm.setFixedWidth(28); rm.clicked.connect(lambda: self.removed.emit(self))
            h.addWidget(self.chk); h.addWidget(self.edit, 1); h.addWidget(rm); v.addLayout(h)
            self.prev = W.QLabel(); self.prev.setStyleSheet(WHITE + "min-height:30px;"); v.addWidget(self.prev)
            h2 = W.QHBoxLayout(); cb = W.QPushButton("色"); cb.setFixedWidth(40); cb.clicked.connect(self.pick)
            self.lw = W.QDoubleSpinBox(); self.lw.setRange(0.5, 12); self.lw.setSingleStep(0.5); self.lw.setValue(lw)
            self.lw.setPrefix("太さ "); self.lw.valueChanged.connect(self.changed)
            h2.addWidget(cb); h2.addWidget(self.lw); h2.addStretch(1); v.addLayout(h2)
            self.err = W.QLabel(); self.err.setStyleSheet("color:#c00;"); self.err.setWordWrap(True); v.addWidget(self.err)
            self.style_btn(); self.set_active(False); self.update_preview()

        def text(self): return self.edit.text()

        def style_btn(self):
            self.chk.setStyleSheet(f"border-radius:13px; border:2px solid {self.color}; "
                                   f"background:{self.color if self.chk.isChecked() else '#eee'};")

        def set_active(self, a): self.setStyleSheet("#row{border:2px solid %s;}" % (self.color if a else "#ccc"))

        def pick(self):
            c = W.QColorDialog.getColor(QtGui.QColor(self.color), self)
            if c.isValid(): self.color = c.name(); self.style_btn(); self.changed.emit()

        def on_edit(self): self.update_preview(); self.changed.emit()

        def update_preview(self):
            t_ = self.edit.text()
            if not t_.strip(): self.prev.clear(); self.err.setText(""); return
            try:
                self.pre(); tex = tex_item(t_); pm = tex_pixmap(tex, 125)
                self.prev.setPixmap(pm) if pm else self.prev.setText(tex); self.err.setText("")
            except Exception as ex:
                self.prev.clear(); self.err.setText(f"入力エラー: {ex}")

    class KeyDialog(W.QDialog):
        """キーパッド設定: ページの追加・削除・並べ替え・名前変更、キーの追加・編集"""
        def __init__(self, parent=None):
            super().__init__(parent); self.setWindowTitle("キーパッド設定"); self.resize(780, 500)
            self.pages = json.loads(json.dumps(KEYPAGES)); self.busy = False
            lay = W.QHBoxLayout(self); lv = W.QVBoxLayout(); lv.addWidget(W.QLabel("ページ (ダブルクリックで名前変更)"))
            self.plist = W.QListWidget(); lv.addWidget(self.plist, 1); lay.addLayout(lv, 1)
            rv = W.QVBoxLayout(); rv.addWidget(W.QLabel("キー:  表示名 / 挿入テキスト  ( | = カーソル位置。選択中の文字は | の位置に包み込みます)"))
            self.table = W.QTableWidget(0, 2); self.table.setHorizontalHeaderLabels(["表示名", "挿入テキスト"])
            self.table.horizontalHeader().setStretchLastSection(True); rv.addWidget(self.table, 1); lay.addLayout(rv, 2)

            def btns(box, items):
                for txt, fn in items:
                    b = W.QPushButton(txt); b.clicked.connect(lambda _=False, f=fn: f()); box.addWidget(b)
            h1 = W.QHBoxLayout(); btns(h1, [("追加", self.add_page), ("削除", self.del_page), ("↑", lambda: self.mv_page(-1)), ("↓", lambda: self.mv_page(1))]); lv.addLayout(h1)
            h2 = W.QHBoxLayout(); btns(h2, [("キー追加", self.add_key), ("キー削除", self.del_key), ("↑", lambda: self.mv_key(-1)), ("↓", lambda: self.mv_key(1))]); rv.addLayout(h2)
            h3 = W.QHBoxLayout(); btns(h3, [("既定に戻す", self.reset), ("OK", self.accept), ("キャンセル", self.reject)]); rv.addLayout(h3)
            self.plist.itemChanged.connect(self.rename); self.plist.currentRowChanged.connect(self.load)
            self.table.cellChanged.connect(self.commit); self.refresh(0)

        def refresh(self, sel):
            self.busy = True; self.plist.clear()
            for p in self.pages:
                it = W.QListWidgetItem(p["name"]); it.setFlags(it.flags() | Qt.ItemIsEditable); self.plist.addItem(it)
            self.busy = False; self.plist.setCurrentRow(max(0, min(sel, len(self.pages) - 1))); self.load(self.plist.currentRow())

        def load(self, i):
            if self.busy: return
            self.busy = True; self.table.setRowCount(0)
            if 0 <= i < len(self.pages):
                for lab, txt in self.pages[i]["keys"]:
                    r = self.table.rowCount(); self.table.insertRow(r)
                    self.table.setItem(r, 0, W.QTableWidgetItem(lab)); self.table.setItem(r, 1, W.QTableWidgetItem(txt))
            self.busy = False

        def cur(self): return self.plist.currentRow()
        def rename(self, it):
            if not self.busy: self.pages[self.plist.row(it)]["name"] = it.text()

        def commit(self, *_):
            if self.busy or self.cur() < 0: return
            g = lambda r, c: self.table.item(r, c).text() if self.table.item(r, c) else ""
            self.pages[self.cur()]["keys"] = [[g(r, 0), g(r, 1)] for r in range(self.table.rowCount())]

        def add_page(self): self.commit(); self.pages.append({"name": "新しいページ", "keys": []}); self.refresh(len(self.pages) - 1)
        def del_page(self):
            if len(self.pages) > 1 and self.cur() >= 0: del self.pages[self.cur()]; self.refresh(0)

        def mv_page(self, d):
            i = self.cur(); j = i + d; self.commit()
            if 0 <= j < len(self.pages) and i >= 0: self.pages[i], self.pages[j] = self.pages[j], self.pages[i]; self.refresh(j)

        def add_key(self): self.commit(); self.pages[self.cur()]["keys"].append(["新", ""]); self.load(self.cur())
        def del_key(self):
            r = self.table.currentRow(); self.commit()
            if r >= 0: del self.pages[self.cur()]["keys"][r]; self.load(self.cur())

        def mv_key(self, d):
            r = self.table.currentRow(); self.commit(); ks = self.pages[self.cur()]["keys"]
            if 0 <= r and 0 <= r + d < len(ks):
                ks[r], ks[r + d] = ks[r + d], ks[r]; self.load(self.cur()); self.table.selectRow(r + d)

        def reset(self): self.pages = json.loads(json.dumps(DEFAULT_KEYS)); self.refresh(0)
        def accept(self):
            self.commit(); KEYPAGES[:] = self.pages; save_keys(); BUS.changed.emit(); super().accept()

    class ParamRow(W.QWidget):
        """パラメータ1つ: 名前 / 実数・整数 / 範囲(数値や π などの定数式) / スライダー / 再生。式の中で変数として使える"""
        changed = QtCore.Signal(); moved = QtCore.Signal(); playToggled = QtCore.Signal(); removed = QtCore.Signal(object)

        def __init__(self, name="t", kind="real", lo=0.0, hi=6.28, val=0.0, fixed=False):
            super().__init__(); self.fixed = fixed; self._lov, self._hiv = 0.0, 1.0
            h = W.QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(3)
            self.name = W.QLineEdit(name); self.name.setFixedWidth(48); self.name.setReadOnly(fixed); self.name.setToolTip("パラメータ名")
            self.kind = W.QComboBox(); self.kind.addItems(["実数", "整数"]); self.kind.setCurrentIndex(1 if kind == "int" else 0)
            self.lo = W.QLineEdit(lo if isinstance(lo, str) else f"{lo:g}"); self.hi = W.QLineEdit(hi if isinstance(hi, str) else f"{hi:g}")
            for bx in (self.lo, self.hi): bx.setFixedWidth(76); bx.setToolTip("数値または定数式 (例: pi, 2π, -pi/2, sqrt(2))")
            self.reparse()
            self.slider = W.QSlider(Qt.Horizontal); self.slider.setRange(0, 400)
            self.slider.setValue(int(round(400 * (min(max(val, self._lov), self._hiv) - self._lov) / ((self._hiv - self._lov) or 1))))
            self.lab = W.QLabel(); self.lab.setMinimumWidth(70)
            self.play = W.QPushButton("▶"); self.play.setCheckable(True); self.play.setFixedWidth(28)
            self.rm = W.QPushButton("×"); self.rm.setFixedWidth(24); self.rm.setVisible(not fixed)
            for w in (self.name, self.kind, self.lo, self.slider, self.hi, self.lab, self.play, self.rm): h.addWidget(w, 1 if w is self.slider else 0)
            self.name.editingFinished.connect(self.changed); self.kind.currentIndexChanged.connect(lambda _: self.changed.emit())
            self.lo.editingFinished.connect(self._edited); self.hi.editingFinished.connect(self._edited)
            self.changed.connect(lambda: self.lab.setText(f"= {self.value():g}"))
            self.slider.valueChanged.connect(self._moved); self.play.toggled.connect(lambda _: self.playToggled.emit())
            self.rm.clicked.connect(lambda: self.removed.emit(self)); self._moved()

        def reparse(self):
            """範囲欄の式を数値化 (不正なら赤表示にして直前の有効な値を保持)"""
            for ed, attr in ((self.lo, "_lov"), (self.hi, "_hiv")):
                try: setattr(self, attr, num_of(ed.text())); ed.setStyleSheet("")
                except Exception: ed.setStyleSheet("background:#fcc;")

        def _edited(self): self.reparse(); self.changed.emit()
        def lo_text(self): return self.lo.text()
        def hi_text(self): return self.hi.text()
        def kindname(self): return "int" if self.kind.currentIndex() == 1 else "real"
        def range(self): a, b = self._lov, self._hiv; return (min(a, b), max(a, b) if a != b else a + 1e-9)

        def value(self):
            a, b = self._lov, self._hiv; v = a + (b - a) * self.slider.value() / 400
            return float(round(v)) if self.kindname() == "int" else v

        def _moved(self, *_): self.lab.setText(f"= {self.value():g}"); self.moved.emit()
        def advance(self): self.slider.setValue((self.slider.value() + 1) % 401)

    class Tb(Toolbar):
        """画像保存ボタンを外し、代わりにこのページの式をJSONで保存/読込するボタンを付けたツールバー"""
        toolitems = [t_ for t_ in Toolbar.toolitems if t_[0] != "Save"]

        def __init__(self, canvas, panel):
            super().__init__(canvas, panel); self.panel = panel; self.addSeparator()
            self.addAction("式を保存(JSON)").triggered.connect(lambda: panel.save_panel())
            self.addAction("式を読込").triggered.connect(lambda: panel.load_panel())

        def save_figure(self, *a): self.panel.save_panel()

    class Panel(W.QWidget):
        """1つのグラフ。左(約3): 入力・パレット / 右(約7): グラフ。マウス: ホイール=拡大縮小, 左ドラッグ=移動, 右ドラッグ=回転(3D)"""
        closeRequested = QtCore.Signal(object); fullRequested = QtCore.Signal(object)

        def __init__(self, space="xy平面"):
            super().__init__(); self.rows = []; self.space = None; self._view = None; self.ax = None
            self.drag = None; self.lim = [-5.0, 5.0] * 3; self.zoom3d = 1.0; self.last_edit = None; self.kp = 0; self._defsig = None; self.def_errs = {}
            self.params = []; self._pnames = ["t"]; self._fit = None
            main = W.QHBoxLayout(self); main.setContentsMargins(0, 0, 0, 0)
            split = W.QSplitter(Qt.Horizontal); main.addWidget(split)
            # ---------- 左: 入力 ----------
            left = W.QWidget(); ll = W.QVBoxLayout(left); ll.setContentsMargins(4, 4, 4, 4)
            top = W.QHBoxLayout(); self.spaceBox = W.QComboBox(); self.spaceBox.addItems(list(SPACES))
            fb = W.QPushButton("⛶"); fb.setFixedWidth(28); fb.setToolTip("全画面表示 (Escで戻る)")
            fb.clicked.connect(lambda: self.fullRequested.emit(self))
            xb = W.QPushButton("×"); xb.setFixedWidth(28); xb.setToolTip("このパネルを閉じる")
            xb.clicked.connect(lambda: self.closeRequested.emit(self))
            top.addWidget(W.QLabel("空間:")); top.addWidget(self.spaceBox, 1); top.addWidget(fb); top.addWidget(xb)
            ll.addLayout(top)
            self.axpW = W.QWidget(); ah = W.QHBoxLayout(self.axpW); ah.setContentsMargins(0, 0, 0, 0)
            self.axpBox = W.QComboBox(); self.axpBox.addItem("t"); self.axpBox.currentIndexChanged.connect(lambda _: self.redraw())
            ah.addWidget(W.QLabel("第3軸にするパラメータ:")); ah.addWidget(self.axpBox, 1); ll.addWidget(self.axpW)
            self.listw = W.QWidget(); self.listl = W.QVBoxLayout(self.listw); self.listl.setContentsMargins(0, 0, 0, 0)
            self.listl.addStretch(1); sa = W.QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(self.listw)
            ll.addWidget(sa, 1)
            addb = W.QPushButton("＋ 式を追加"); addb.clicked.connect(lambda: self.add_row(focus=True)); ll.addWidget(addb)
            self.cedit = W.QLineEdit(); self.cedit.textChanged.connect(self.on_consts)
            self.cedit.setPlaceholderText("定数 例: n=2, a=pi/3 (確定した値は近似値も表示)"); ll.addWidget(self.cedit)
            kb = W.QGroupBox("キーパッド"); kl = W.QVBoxLayout(kb); nav = W.QHBoxLayout()
            for txt, fn in (("◀", lambda: self.set_page(self.kp - 1)), ("▶", lambda: self.set_page(self.kp + 1))):
                b = W.QPushButton(txt); b.setFixedWidth(30); b.setFocusPolicy(Qt.NoFocus); b.clicked.connect(lambda _=False, f=fn: f())
                nav.addWidget(b)
                if txt == "◀": self.pageBox = W.QComboBox(); self.pageBox.setFocusPolicy(Qt.NoFocus); nav.addWidget(self.pageBox, 1)
            self.pageBox.setSizeAdjustPolicy(W.QComboBox.AdjustToMinimumContentsLengthWithIcon); self.pageBox.setMinimumContentsLength(6)
            self.pageBox.activated.connect(self.set_page)
            st = W.QPushButton("⚙"); st.setFixedWidth(30); st.setFocusPolicy(Qt.NoFocus); st.setToolTip("キーパッド設定")
            st.clicked.connect(lambda: KeyDialog(self).exec()); nav.addWidget(st); kl.addLayout(nav)
            self.kscroll = W.QScrollArea(); self.kscroll.setWidgetResizable(True); self.kscroll.setFixedHeight(4 * 36 + 8)
            self.kscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); kl.addWidget(self.kscroll); ctl = W.QHBoxLayout()
            for lab, act in (("⌫", "bs"), ("←", "l"), ("→", "r"), ("全消去", "clear"), ("＋式", "new")):
                b = W.QToolButton(); b.setText(lab); b.setFocusPolicy(Qt.NoFocus); b.setMinimumHeight(30)
                b.clicked.connect(lambda _=False, a=act: self.key_action(a)); ctl.addWidget(b)
            kl.addLayout(ctl); ll.addWidget(kb)
            grp = W.QGroupBox("軸・スケール"); gv = W.QVBoxLayout(grp); axw = W.QWidget(); gl = W.QGridLayout(axw)
            gv.addWidget(axw); grp.setCheckable(True); grp.setChecked(False); axw.setVisible(False); grp.toggled.connect(axw.setVisible)
            self.axrows = []
            for i in range(3):
                lab, lo, tl, hi = W.QLabel(), W.QDoubleSpinBox(), W.QLabel("〜"), W.QDoubleSpinBox()
                for bx in (lo, hi):
                    bx.setRange(-1e6, 1e6); bx.setDecimals(3); bx.setKeyboardTracking(False); bx.valueChanged.connect(self.on_axis)
                for j, w in enumerate((lab, lo, tl, hi)): gl.addWidget(w, i, j)
                self.axrows.append((lab, lo, tl, hi))
            self.equalChk = W.QCheckBox("縦横(各軸)を等倍にする"); self.equalChk.toggled.connect(self.redraw)
            gl.addWidget(self.equalChk, 3, 0, 1, 4); ll.addWidget(grp)
            self.leftw = left; left.setMinimumWidth(300); split.addWidget(left)
            W.QApplication.instance().focusChanged.connect(self.on_focus); BUS.changed.connect(self.build_keypad); self.build_keypad()
            # ---------- 右: グラフ ----------
            right = W.QWidget(); rl = W.QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
            self.fig = Figure(facecolor="white"); self.canvas = Canvas(self.fig); self.tb = Tb(self.canvas, self)
            self.canvas.setSizePolicy(W.QSizePolicy.Expanding, W.QSizePolicy.Expanding)
            rl.addWidget(self.tb); rl.addWidget(self.canvas, 1)
            pb = W.QWidget(); pbl = W.QVBoxLayout(pb); pbl.setContentsMargins(4, 2, 4, 2); pbl.setSpacing(1)
            self.plist = W.QVBoxLayout(); pbl.addLayout(self.plist); hb = W.QHBoxLayout()
            addp = W.QPushButton("＋ パラメータ"); addp.clicked.connect(lambda: self.add_param()); hb.addWidget(addp); hb.addStretch(1); pbl.addLayout(hb)
            psa = W.QScrollArea(); psa.setWidgetResizable(True); psa.setWidget(pb); psa.setFixedHeight(104)
            self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.step)
            self.add_param("t", "real", 0.0, 6.28, 0.0, fixed=True, notify=False)
            rl.addWidget(psa); split.addWidget(right)
            split.setStretchFactor(0, 3); split.setStretchFactor(1, 7); split.setSizes([300, 700])
            c = self.canvas
            c.mpl_connect("scroll_event", self.on_scroll); c.mpl_connect("button_press_event", self.on_press)
            c.mpl_connect("motion_notify_event", self.on_move); c.mpl_connect("button_release_event", lambda e: setattr(self, "drag", None))
            c.mpl_connect("resize_event", lambda e: self.redraw())
            self.redraw_timer = QtCore.QTimer(self); self.redraw_timer.setSingleShot(True); self.redraw_timer.timeout.connect(self.draw)
            self.spaceBox.currentTextChanged.connect(self.set_space)
            self.spaceBox.setCurrentText(space); self.set_space(space)

        def set_input_visible(self, on): self.leftw.setVisible(on)

        # --- キーパッド ---
        def build_keypad(self):
            pages = KEYPAGES + [{"name": "パラメータ", "keys": [[n_, n_] for n_ in self._pnames]}]   # 末尾のページは現在のパラメータから自動生成
            n = len(pages); self.kp = max(0, min(self.kp, n - 1))
            self.pageBox.blockSignals(True); self.pageBox.clear(); self.pageBox.addItems([p["name"] for p in pages])
            self.pageBox.setCurrentIndex(self.kp); self.pageBox.blockSignals(False)
            box = W.QWidget(); g = W.QGridLayout(box); g.setContentsMargins(2, 2, 2, 2); g.setSpacing(2); keys = pages[self.kp]["keys"]
            for j, (lab, text) in enumerate(keys):
                bt = W.QToolButton(); bt.setText(lab); bt.setFocusPolicy(Qt.NoFocus); bt.setFixedSize(40, 32)
                bt.setToolTip(text); bt.clicked.connect(lambda _=False, tx=text: self.insert(tx)); g.addWidget(bt, j // 7, j % 7)
            g.setRowStretch(len(keys) // 7 + 1, 1); g.setColumnStretch(7, 1)      # 全ページ共通の固定サイズ。はみ出す分は内側だけスクロール
            old = self.kscroll.takeWidget(); self.kscroll.setWidget(box)
            if old is not None: old.deleteLater()

        def set_page(self, i): self.kp = i % (len(KEYPAGES) + 1); self.build_keypad()

        def on_focus(self, old, new):
            for r in self.rows: r.set_active(new is r.edit)
            if any(new is r.edit for r in self.rows): self.last_edit = new

        def target_edit(self):
            e = W.QApplication.focusWidget()
            if any(e is r.edit for r in self.rows): return e
            if self.last_edit is not None and any(self.last_edit is r.edit for r in self.rows): return self.last_edit
            return self.rows[-1].edit if self.rows else None

        def insert(self, text):
            """カーソル位置にキーを挿入 (| の位置にカーソルを置く。選択中の文字は | の位置に包む)"""
            e = self.target_edit()
            if e is None: self.add_row(focus=True); e = self.rows[-1].edit
            i = text.find("|"); t_ = text.replace("|", ""); sel = e.selectedText() if i >= 0 else ""
            if sel: t_ = t_[:i] + sel + t_[i:]
            e.insert(t_)
            if i >= 0: e.setCursorPosition(e.cursorPosition() - (len(t_) - i - len(sel)))
            e.setFocus()

        def key_action(self, name):
            if name == "new": return self.add_row(focus=True)
            e = self.target_edit()
            if e is None: return
            if name == "bs": e.backspace()
            elif name == "l": e.setCursorPosition(max(0, e.cursorPosition() - 1))
            elif name == "r": e.setCursorPosition(e.cursorPosition() + 1)
            elif name == "clear": e.clear()
            e.setFocus()

        # --- 空間の切り替え / 式の追加 ---
        def set_space(self, name, defaults=True):
            self.space = name; S = SPACES[name]; self._view = None; self.lim = [-5.0, 5.0] * 3
            self.equalChk.setChecked(S["equal"])
            for r in self.rows: r.setParent(None); r.deleteLater()
            self.rows = []
            for e in (S["defaults"] if defaults else []): self.add_row(e, redraw=False)
            if defaults and not self.rows: self.add_row("", redraw=False)
            self.sync_spins(); self.redraw()

        def add_row(self, text="", color=None, on=True, lw=2.0, redraw=True, focus=False):
            r = ObjRow(text, self.apply_consts, color, on, lw)
            r.changed.connect(self.redraw); r.removed.connect(self.remove_row); r.enter.connect(lambda _: self.add_row(focus=True))
            self.listl.insertWidget(len(self.rows), r); self.rows.append(r)
            if focus: r.edit.setFocus()
            if redraw: self.redraw()

        def remove_row(self, r):
            self.rows.remove(r); r.setParent(None); r.deleteLater(); self.redraw()

        # --- 軸 ---
        def sync_spins(self):
            S = SPACES[self.space]; self.axpW.setVisible(bool(S.get("tz")))
            for i, row in enumerate(self.axrows):
                show = i < 2 or (S["dim"] == 3 and not S.get("tz"))
                for w in row: w.setVisible(show)
                row[0].setText(S["labels"][i] + " :" if i < len(S["labels"]) else "")
                for bx, v in ((row[1], self.lim[2 * i]), (row[3], self.lim[2 * i + 1])):
                    bx.blockSignals(True); bx.setValue(v); bx.blockSignals(False)

        def on_axis(self, *_):
            for i, (_, lo, _, hi) in enumerate(self.axrows):
                if lo.value() < hi.value(): self.lim[2 * i], self.lim[2 * i + 1] = lo.value(), hi.value()
            self.redraw()

        def fit_equal(self):
            w, h = self.canvas.width() * 0.94, self.canvas.height() * 0.93
            if w < 50 or h < 50: return
            L = self.lim; u = max((L[1] - L[0]) / w, (L[3] - L[2]) / h)
            cx, cy = (L[0] + L[1]) / 2, (L[2] + L[3]) / 2
            L[:4] = [cx - u * w / 2, cx + u * w / 2, cy - u * h / 2, cy + u * h / 2]; self.sync_spins()

        # --- マウス操作 ---
        def on_scroll(self, e):
            if self.ax is None or self.tb.mode: return
            if e.key == "control" and SPACES[self.space]["dim"] == 3:   # Ctrl+ホイール: 3D表示の拡大率
                self.zoom3d = max(0.3, self.zoom3d * (1.1 if e.button == "up" else 1 / 1.1)); self.redraw(); return
            S = SPACES[self.space]; f = 0.85 if e.button == "up" else 1 / 0.85; L = self.lim
            n = 2 if S["dim"] == 2 or S.get("tz") else 3
            for i in range(n):
                cur = (e.xdata, e.ydata)[i] if S["dim"] == 2 and e.xdata is not None else None
                c = (L[2 * i] + L[2 * i + 1]) / 2 if cur is None else cur
                L[2 * i:2 * i + 2] = [c + (v - c) * f for v in L[2 * i:2 * i + 2]]
            self.sync_spins(); self.redraw()

        def on_press(self, e):
            if self.ax is not None and not self.tb.mode and e.button in (1, 3):
                self.drag = (e.button, e.x, e.y, list(self.lim), (getattr(self.ax, "elev", 0), getattr(self.ax, "azim", 0)))

        def on_move(self, e):
            if not self.drag or e.x is None: return
            btn, x0, y0, L0, (el0, az0) = self.drag; S = SPACES[self.space]; dx, dy = e.x - x0, e.y - y0
            bb = self.ax.bbox; fx, fy = dx / bb.width, dy / bb.height
            if S["dim"] == 2 and btn in (1, 3):      # 2D: ドラッグで移動
                for i, f in ((0, fx), (1, fy)):
                    sp_ = L0[2 * i + 1] - L0[2 * i]; self.lim[2 * i:2 * i + 2] = [L0[2 * i] - f * sp_, L0[2 * i + 1] - f * sp_]
            elif btn == 3:                            # 3D: 右ドラッグで回転 (再計算なし)
                self._view = (max(-90, min(90, el0 + dy * 0.4)), az0 - dx * 0.4)
                self.ax.view_init(*self._view); self.refit3d(); self.canvas.draw_idle(); return
            elif btn == 1:                            # 3D: 左ドラッグで画面平行移動
                az, el = np.radians(az0), np.radians(el0)
                right = (-np.sin(az), np.cos(az), 0); up = (-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el))
                for i in range(2 if S.get("tz") else 3):
                    sp_ = L0[2 * i + 1] - L0[2 * i]; d = -(fx * right[i] + fy * up[i]) * sp_
                    self.lim[2 * i:2 * i + 2] = [L0[2 * i] + d, L0[2 * i + 1] + d]
            self.sync_spins(); self.redraw()

        # --- 定数 / 保存・読込 ---
        def apply_consts(self):
            self.sync_params(); self._consts(); self.def_errs = build_defs([r.text() for r in self.rows], SPACES[self.space]["key"])

        def _consts(self):
            CONSTS.clear()
            for part in re.split(r"[,;]", self.cedit.text()):
                if "=" in part:
                    k, v = part.split("=", 1)
                    try: CONSTS[k.strip()] = parse_expr(v.strip().replace("π", "pi"), local_dict=LOC, transformations=TF)
                    except Exception: pass

        def on_consts(self):
            for r in self.rows: r.update_preview()
            self.redraw()

        def to_dict(self):
            return dict(space=self.space, lim=self.lim, equal=self.equalChk.isChecked(), consts=self.cedit.text(), axp=self.axpBox.currentText(),
                        params=[dict(name=p.name.text(), kind=p.kindname(), lo=p.lo_text(), hi=p.hi_text(), val=p.value(), fixed=p.fixed) for p in self.params],
                        rows=[dict(expr=r.text(), color=r.color, on=r.chk.isChecked(), lw=r.lw.value()) for r in self.rows])

        def from_dict(self, d):
            self.spaceBox.blockSignals(True); self.spaceBox.setCurrentText(d["space"]); self.spaceBox.blockSignals(False)
            self.set_space(d["space"], defaults=False)
            self.lim = list(d["lim"]) if "lim" in d else [-d.get("R", 5.0), d.get("R", 5.0)] * 3
            self.equalChk.setChecked(d.get("equal", self.equalChk.isChecked())); self.cedit.setText(d.get("consts", ""))
            for p in list(self.params): p.setParent(None); p.deleteLater()
            self.params = []
            ps = d.get("params") or [dict(name="t", lo=d.get("tmin", 0.0), hi=d.get("tmax", 6.28), val=0.0, fixed=True)]
            for q in ps: self.add_param(q.get("name"), q.get("kind", "real"), q.get("lo", -5.0), q.get("hi", 5.0), q.get("val", 1.0), q.get("fixed", False), notify=False)
            self.sync_params(); self.axpBox.setCurrentText(d.get("axp", "t"))
            for r in d["rows"]:
                if "expr" in r: self.add_row(r["expr"], r["color"], r["on"], r.get("lw", 2.0), redraw=False)
            self.sync_spins(); self.redraw()

        def save_panel(self):
            p = pick_save(self, "式を保存", "expressions.json")
            if p:
                with open(p, "w", encoding="utf-8") as f: json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

        def load_panel(self):
            p = pick_open(self, "式を読込")
            if p:
                with open(p, encoding="utf-8") as f: d = json.load(f)
                self.from_dict(d["panels"][0] if "panels" in d else d)

        # --- パラメータ ---
        def add_param(self, name=None, kind="real", lo=-5.0, hi=5.0, val=1.0, fixed=False, notify=True):
            if name is None: name = next((c for c in "abcdfghpquvw" if c not in [p.name.text() for p in self.params]), "p")
            p = ParamRow(name, kind, lo, hi, val, fixed)
            p.changed.connect(self.on_params); p.moved.connect(self.redraw); p.playToggled.connect(self.update_timer); p.removed.connect(self.remove_param)
            self.plist.addWidget(p); self.params.append(p)
            if notify: self.on_params()

        def remove_param(self, p):
            self.params.remove(p); p.setParent(None); p.deleteLater(); self.update_timer(); self.on_params()

        def update_timer(self): self.timer.start(30) if any(p.play.isChecked() for p in self.params) else self.timer.stop()

        def step(self):
            for p in self.params:
                if p.play.isChecked(): p.advance()

        def on_params(self):
            self.sync_params()
            for r in self.rows: r.update_preview()
            self.redraw()

        def sync_params(self):
            """パラメータ名・現在値を式の評価エンジンへ反映 (不正な名前は赤表示で無視)"""
            names, seen = [], {"t"}
            for p in self.params:
                p.reparse(); n = p.name.text().strip()
                ok = p.fixed or bool(re.fullmatch(r"[^\W\d]\w*", n) and n not in seen and not param_reserved(n))
                p.name.setStyleSheet("" if ok else "background:#fcc;")
                if ok:
                    seen.add(n); PVAL[n] = p.value()
                    if not p.fixed: names.append(n)
            set_params(names); allp = ["t"] + names
            if allp != self._pnames:
                self._pnames = allp; cur = self.axpBox.currentText()
                self.axpBox.blockSignals(True); self.axpBox.clear(); self.axpBox.addItems(allp)
                self.axpBox.setCurrentText(cur if cur in allp else "t"); self.axpBox.blockSignals(False); self.build_keypad()

        def refit3d(self):
            if self._fit and self.ax is not None: fit3d(self.ax, self.fig, *self._fit)

        # --- 描画 (連続操作中も一定間隔で随時更新) ---
        def redraw(self, *_):
            if not self.redraw_timer.isActive(): self.redraw_timer.start(20)

        def draw(self):
            S = SPACES[self.space]; self.apply_consts(); dim = S["dim"]; tz = S.get("tz")
            tname = (self.axpBox.currentText() or "t") if tz else "t"; AX[0] = tname      # 第3軸にするパラメータ
            prow = next((p for p in self.params if p.name.text().strip() == tname), self.params[0]); t = prow.value(); tr = prow.range()
            if tz: self.lim[4], self.lim[5] = tr
            if self.ax is not None and hasattr(self.ax, "azim"): self._view = (self.ax.elev, self.ax.azim)
            if dim == 2 and self.equalChk.isChecked(): self.fit_equal()
            L = self.lim; R = max(abs(v) for v in (L if dim == 3 and not tz else L[:4]))
            self.fig.clear()
            ax = self.ax = self.fig.add_axes([0.055, 0.065, 0.94, 0.93] if dim == 2 else [0, 0, 1, 1],
                                             projection="3d" if dim == 3 else None)
            if dim == 3: getattr(ax, "disable_mouse_rotation", lambda: None)()
            ctx = SimpleNamespace(R=R, t=t, tr=tr, cond="", lw=2.0, maps=[])
            if S["key"] == "cx":   # w=f(z) の行を集める
                for r in self.rows:
                    mm = re.match(MAP_RE, split_cond(r.text())[0])
                    if mm and r.chk.isChecked():
                        try: ctx.maps.append(compile_expr(mm.group(1))[2])
                        except Exception: pass
            sig = str(sorted(map(str, DEFS.items())))
            if sig != self._defsig:   # 定義が変わったら他の式のプレビューも更新
                self._defsig = sig
                for r in self.rows: r.update_preview()
            for i, r in enumerate(self.rows):
                if self.def_errs.get(i): r.err.setText("定義エラー: " + self.def_errs[i]); continue
                if not r.chk.isChecked() or not r.text().strip(): continue
                try:
                    ctx.lw = r.lw.value()
                    with matplotlib.rc_context({"lines.linewidth": ctx.lw}):
                        draw_item(ax, r.text(), S, ctx, r.color)
                    r.err.setText("")
                except Exception as ex:
                    r.err.setText(f"描画エラー: {ex}")
            if S.get("polar"):   # 極座標グリッド
                th_ = np.linspace(0, 2 * np.pi, 200)
                for k in range(1, 40): ax.plot(k * R / 5 * np.cos(th_), k * R / 5 * np.sin(th_), color="gray", lw=0.5, alpha=0.4)
                for a_ in np.arange(0, 2 * np.pi, np.pi / 6):
                    ax.plot([0, 20 * R * np.cos(a_)], [0, 20 * R * np.sin(a_)], color="gray", lw=0.5, alpha=0.4)
            lb = list(S["labels"])
            if tz: lb[2] = tname
            if dim == 2:
                axes2d(ax, L, lb, max(4, int(round(10 * self.canvas.height() / max(self.canvas.width(), 1)))))
            else:
                ax.set_xlim3d(L[0], L[1]); ax.set_ylim3d(L[2], L[3]); ax.set_zlim3d(L[4], L[5]); axes3d(ax, L, lb)
                cw, ch = max(self.canvas.width(), 1), max(self.canvas.height(), 1); kx = min(2.2, max(1.0, cw / ch)) ** 0.8
                asp = [L[1] - L[0], L[3] - L[2], L[5] - L[4]] if self.equalChk.isChecked() else [kx, kx, 1]
                if self._view: ax.view_init(*self._view)
                self._fit = (L, tuple(v / max(asp) for v in asp), 0.96 * self.zoom3d); self.refit3d()   # 表示欄いっぱいに拡大
            self.canvas.draw_idle()

    class GraphTab(W.QSplitter):
        """1タブ = 複数パネルを並べられる分割領域"""
        def __init__(self, space="xy平面", data=None):
            super().__init__(Qt.Horizontal); self.full = None
            QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, self.exit_full)
            if data:
                for pd in data["panels"]: self.add_panel(data=pd)
            else: self.add_panel(space)

        def add_panel(self, space="xy平面", data=None):
            p = Panel(space)
            if data: p.from_dict(data)
            p.closeRequested.connect(self.remove_panel); p.fullRequested.connect(self.toggle_full); self.addWidget(p)

        def remove_panel(self, p):
            if self.count() > 1: self.exit_full(); p.setParent(None); p.deleteLater()

        def toggle_full(self, p): self.exit_full() if self.full else self.enter_full(p)

        def enter_full(self, p):   # 選んだパネルだけを全画面表示
            w = self.window(); self.full = p
            for i in range(self.count()): self.widget(i).setVisible(self.widget(i) is p)
            p.set_input_visible(False); w.tabs.tabBar().hide(); w.menuBar().hide(); w.showFullScreen()

        def exit_full(self):
            if not self.full: return
            w = self.window(); self.full.set_input_visible(True); self.full = None
            for i in range(self.count()): self.widget(i).show()
            w.tabs.tabBar().show(); w.menuBar().show(); w.showNormal()

        def to_dict(self): return {"panels": [self.widget(i).to_dict() for i in range(self.count())]}

        def toggle_orientation(self):
            self.setOrientation(Qt.Vertical if self.orientation() == Qt.Horizontal else Qt.Horizontal)

    WINDOWS = []

    class MainWindow(W.QMainWindow):
        count = 0

        def __init__(self):
            super().__init__(); self.setWindowTitle("MathPlot Studio"); self.resize(*SETTINGS["window_size"])
            self.tabs = W.QTabWidget(); self.tabs.setTabsClosable(True); self.tabs.setMovable(True)
            self.tabs.tabCloseRequested.connect(self.close_tab); self.setCentralWidget(self.tabs)
            m = self.menuBar().addMenu("ファイル")
            m.addAction("新しいタブ", self.new_tab, "Ctrl+T"); m.addAction("新しいウィンドウ", new_window, "Ctrl+N")
            m.addAction("キーパッド設定…", lambda: KeyDialog(self).exec())
            m.addAction("保存…", self.save, "Ctrl+S"); m.addAction("開く…", self.open_file, "Ctrl+O")
            m.addAction("タブを閉じる", lambda: self.close_tab(self.tabs.currentIndex()), "Ctrl+W")
            g = self.menuBar().addMenu("グラフ")
            g.addAction("パネルを追加 (同じタブ内に並べる)", lambda: self.cur().add_panel() if self.cur() else None, "Ctrl+D")
            g.addAction("パネルの並び方向を切替", lambda: self.cur().toggle_orientation() if self.cur() else None)
            self.new_tab()

        def closeEvent(self, e):
            if not self.isFullScreen() and not self.isMaximized():
                SETTINGS["window_size"] = [self.width(), self.height()]; save_settings()
            super().closeEvent(e)

        def cur(self): return self.tabs.currentWidget()

        def new_tab(self, data=None):
            MainWindow.count += 1
            i = self.tabs.addTab(GraphTab(SETTINGS["default_space"] if SETTINGS.get("default_space") in SPACES else "xy平面", data=data if isinstance(data, dict) else None), f"グラフ {MainWindow.count}"); self.tabs.setCurrentIndex(i)

        def save(self):
            p = pick_save(self, "保存", "graph.json")
            if p:
                with open(p, "w", encoding="utf-8") as f: json.dump(self.cur().to_dict(), f, ensure_ascii=False, indent=1)

        def open_file(self):
            p = pick_open(self, "開く")
            if p:
                with open(p, encoding="utf-8") as f: self.new_tab(json.load(f))

        def close_tab(self, i):
            w = self.tabs.widget(i); self.tabs.removeTab(i); w.deleteLater()
            if self.tabs.count() == 0: self.close()

    def new_window():
        w = MainWindow(); WINDOWS.append(w); w.show(); return w

    def _detach():
        """コンソール/VSCodeを閉じても終了しないよう、独立したプロセスとして再起動する (--no-detach で無効)"""
        if getattr(sys, "frozen", False) or "--child" in sys.argv or "--no-detach" in sys.argv or not SETTINGS.get("detach", True): return False
        import subprocess
        exe = sys.executable; cmd = [exe, os.path.abspath(__file__), "--child"] + sys.argv[1:]; kw = {}
        if os.name == "nt":
            pw = os.path.join(os.path.dirname(exe), "pythonw.exe"); cmd[0] = pw if os.path.exists(pw) else exe
            flag_sets = [0x08 | 0x200 | 0x01000000, 0x08 | 0x200]       # DETACHED | NEW_PROCESS_GROUP (| BREAKAWAY_FROM_JOB)
        else: flag_sets = [None]; kw["start_new_session"] = True
        for fl in flag_sets:
            try:
                k2 = dict(kw)
                if fl is not None: k2["creationflags"] = fl
                subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **k2); return True
            except OSError: continue
        return False

    if __name__ == "__main__":
        if _detach(): sys.exit(0)
        if sys.stderr is None or sys.stdout is None:   # pythonw ではコンソールが無いのでログファイルへ
            sys.stderr = sys.stdout = open(cfg_path("mathplot_studio.log"), "a", encoding="utf-8")
        sys.argv = [a for a in sys.argv if a not in ("--child", "--no-detach")]
        app = W.QApplication(sys.argv); new_window(); sys.exit(app.exec())
