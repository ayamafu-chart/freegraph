"""FreeGraph - 数式グラフ描画 (Streamlit版)。iPad の Chrome など、タッチ操作のブラウザ向け

構成:  app.py (この画面) → gesture_view.py (指の操作) / mathplot_web.py (橋渡し) → mathplot_core.py (数式の解析・描画)
"""
import base64
import json
import math
import os
import re

import streamlit as st

import mathplot_web as mw
from gesture_view import graph_view

HERE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "FreeGraph"


def load_icon():
    """同じフォルダに icon.png (jpg / webp / ico も可) があれば、正方形に整えてアイコンとして使う"""
    for n in ("icon.png", "icon.jpg", "icon.jpeg", "icon.webp", "icon.ico", "icon.gif"):
        p = os.path.join(HERE, n)
        if os.path.exists(p):
            try:
                from PIL import Image
                im = Image.open(p).convert("RGBA")
                w, h = im.size
                m = min(w, h)
                im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((256, 256))
                return im
            except Exception:
                pass
    return None


ICON = load_icon()
st.set_page_config(page_title=APP_NAME, page_icon=ICON if ICON is not None else "📈", layout="wide",
                   initial_sidebar_state="collapsed")

# ======================= 見た目: 元のデスクトップ版に近いシンプルな配色・形 =======================
st.markdown("""
<style>
.stApp {background: #f0f0f0;}
.block-container {padding: 2.2rem 0.8rem 2rem 0.8rem; max-width: 100%;}
header[data-testid="stHeader"] {background: transparent;}
/* iPadは文字が16px未満の入力欄をタップすると勝手に拡大するので 17px にする */
input, textarea, select, [data-baseweb="select"] * {font-size: 17px !important;}
/* ボタン: 角を小さく、淡い灰色。指で押しやすい高さ。ダブルタップ拡大を防ぐ */
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button {
    min-height: 44px; min-width: 44px; touch-action: manipulation; border-radius: 4px;
    background: #fdfdfd; border: 1px solid #c8c8c8; color: #222;}
/* 式のカード・キーパッドの枠: 白 + 細い灰色の枠 */
[data-testid="stVerticalBlockBorderWrapper"] {background: #fafafa; border-radius: 4px; border-color: #b8b8b8;}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {border-radius: 2px;}
[data-testid="stSlider"] [role="slider"] {width: 28px !important; height: 28px !important;}
[data-testid="stSlider"] {padding-top: .1rem; padding-bottom: .3rem;}
[data-testid="stExpander"] {background: #fafafa; border-radius: 4px;}
/* キーパッド・ツールバーは折り返さず横一列 (画面が狭くても崩れない) */
.st-key-keypad [data-testid="stHorizontalBlock"], .st-key-toolbar [data-testid="stHorizontalBlock"],
.st-key-kpnav [data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: .25rem !important;}
.st-key-keypad [data-testid="stColumn"], .st-key-toolbar [data-testid="stColumn"], .st-key-kpnav [data-testid="stColumn"],
.st-key-keypad [data-testid="column"], .st-key-toolbar [data-testid="column"], .st-key-kpnav [data-testid="column"] {
    min-width: 0 !important; flex: 1 1 0 !important; width: auto !important;}
.st-key-keypad button, .st-key-toolbar button, .st-key-kpnav button {width: 100%; padding: 0 !important; font-size: 17px;}
h3 {margin: 0; padding: 0;}
</style>
""", unsafe_allow_html=True)

# iPadのキーボードが数式の先頭を大文字にしたり、自動修正したりしないようにする
try:
    import streamlit.components.v1 as components
    components.html("""
<script>
const w = window.parent;
if (!w.__mpFix) {
  w.__mpFix = true;
  const d = w.document;
  const fix = () => d.querySelectorAll('input[type="text"], textarea').forEach(e => {
    e.setAttribute('autocapitalize', 'off'); e.setAttribute('autocorrect', 'off');
    e.setAttribute('spellcheck', 'false'); e.setAttribute('autocomplete', 'off'); });
  fix(); new MutationObserver(fix).observe(d.body, {childList: true, subtree: true});
}
</script>""", height=0)
except Exception:
    pass

ss = st.session_state


def is_phone():
    """スマホ(iPhone / Androidスマホ)ならTrue。iPad・Androidタブレット・PCはFalse"""
    try:
        ua = str(st.context.headers.get("User-Agent", ""))
    except Exception:
        return False
    return bool(re.search(r"iPhone|iPod|Android.*Mobile|Windows Phone", ua))


# ======================= 状態の管理 =======================
def add_row(text="", color=None, on=True, lw=2.0):
    rid = ss.next_id; ss.next_id += 1
    ss.rows.append({"id": rid})
    ss[f"expr_{rid}"] = text
    ss[f"on_{rid}"] = on
    ss[f"col_{rid}"] = color or mw.PALETTE[rid % len(mw.PALETTE)]
    ss[f"lw_{rid}"] = float(lw)
    ss.active = rid


def add_param(name=None, kind="real", lo="-5", hi="5", val=1.0, fixed=False):
    pid = ss.next_pid; ss.next_pid += 1
    if name is None:
        used = {ss.get(f"pname_{p['id']}") for p in ss.params}
        name = next((c for c in "abcdfghpquvw" if c not in used), "p")
    ss.params.append({"id": pid, "fixed": fixed})
    ss[f"pname_{pid}"] = "t" if fixed else name
    ss[f"pkind_{pid}"] = "整数" if kind == "int" else "実数"
    ss[f"plo_{pid}"] = str(lo)
    ss[f"phi_{pid}"] = str(hi)
    ss[f"pv_{pid}"] = float(val)


def init():
    if "inited" in ss:
        return
    ss.inited = True
    ss.space = ss.space_sel = mw.SPACE_NAMES[0]
    ss.rows, ss.params, ss.next_id, ss.next_pid, ss.active = [], [], 0, 0, None
    add_param("t", "real", "0", "2π", 0.0, fixed=True)
    add_row("")                                   # 最初は空の入力欄が1つ
    for i in range(6):
        ss[f"lim{i}"] = -5.0 if i % 2 == 0 else 5.0
    ss.equal, ss.consts, ss.axp = True, "", "t"
    ss.elev, ss.azim, ss.hfac = 25.0, -60.0, 0.9
    ss.touch = True
    ss.layout = "縦並び" if is_phone() else "横並び"      # スマホは縦並び、iPad/PCは横並び
    ss.kp_base, ss.kp_msg = mw.load_keypages()
    ss.kp_custom = False


def reset_lim():
    for i in range(6):
        ss[f"lim{i}"] = -5.0 if i % 2 == 0 else 5.0


def on_space():
    ss.space = ss.space_sel
    ss.rows = []
    reset_lim()
    add_row("")                                   # 別のグラフに変えたら空の入力欄1つ
    ss.equal = True
    ss.elev, ss.azim = 25.0, -60.0


def get_state():
    """画面の入力をまとめた辞書にする (保存・描画で共通)"""
    rows = [dict(expr=ss.get(f"expr_{r['id']}", ""), on=bool(ss.get(f"on_{r['id']}", True)),
                 color=ss.get(f"col_{r['id']}", "#1f77b4"), lw=float(ss.get(f"lw_{r['id']}", 2.0))) for r in ss.rows]
    params = []
    for p in ss.params:
        i = p["id"]
        params.append(dict(name="t" if p["fixed"] else str(ss.get(f"pname_{i}", "p")),
                           kind="int" if ss.get(f"pkind_{i}") == "整数" else "real",
                           lo=str(ss.get(f"plo_{i}", "0")), hi=str(ss.get(f"phi_{i}", "1")),
                           val=float(ss.get(f"pv_{i}", 0.0)), fixed=p["fixed"]))
    lim = [float(ss.get(f"lim{i}", -5.0 if i % 2 == 0 else 5.0)) for i in range(6)]
    for i in range(3):                      # 最小 >= 最大 の欄は無視
        if lim[2 * i] >= lim[2 * i + 1]:
            lim[2 * i], lim[2 * i + 1] = -5.0, 5.0
    return dict(space=ss.space, rows=rows, params=params, consts=ss.get("consts", ""), lim=lim,
                equal=bool(ss.get("equal", True)), axp=ss.get("axp", "t"),
                elev=float(ss.get("elev", 25.0)), azim=float(ss.get("azim", -60.0)))


def get_state_for_save():
    s = get_state()
    return dict(space=s["space"], lim=s["lim"], equal=s["equal"], consts=s["consts"], axp=s["axp"],
                params=[dict(name=p["name"], kind=p["kind"], lo=p["lo"], hi=p["hi"], val=p["val"], fixed=p["fixed"]) for p in s["params"]],
                rows=s["rows"])


def apply_dict(d):
    """JSON (デスクトップ版と同じ形式) から画面の状態を復元"""
    sp = d.get("space", mw.SPACE_NAMES[0])
    if sp not in mw.SPACES:
        sp = mw.SPACE_NAMES[0]
    ss.space = ss.space_sel = sp
    ss.rows = []
    for r in d.get("rows", []):
        if "expr" in r:
            add_row(r["expr"], r.get("color"), r.get("on", True), r.get("lw", 2.0))
    if not ss.rows:
        add_row("")
    ss.params = []
    ps = d.get("params") or [dict(name="t", lo=0.0, hi=6.28, val=0.0, fixed=True)]
    if not any(q.get("fixed") for q in ps):
        ps = [dict(name="t", lo=0.0, hi=6.28, val=0.0, fixed=True)] + list(ps)
    for q in ps:
        add_param(q.get("name"), q.get("kind", "real"), q.get("lo", -5), q.get("hi", 5), q.get("val", 1.0), q.get("fixed", False))
    lim = d.get("lim") or [-d.get("R", 5.0), d.get("R", 5.0)] * 3
    for i in range(6):
        ss[f"lim{i}"] = float(lim[i]) if i < len(lim) else (-5.0 if i % 2 == 0 else 5.0)
    ss.equal = bool(d.get("equal", True))
    ss.consts = d.get("consts", "")
    ss.axp = d.get("axp", "t")


# --- ボタンから呼ばれる処理 (コールバック: 画面の再描画より前に実行される) ---
def set_active(rid): ss.active = rid
def rm_row(rid): ss.rows = [r for r in ss.rows if r["id"] != rid]
def mark_full(): ss._need_full = True              # 式の入力欄にも関わる変更 → 画面全体を作り直す
def rm_param(pid): ss.params = [p for p in ss.params if p["id"] != pid]; mark_full()
def add_param_cb(): add_param(); mark_full()


def target_row():
    ids = [r["id"] for r in ss.rows]
    if not ids:
        add_row("")
        ids = [ss.rows[0]["id"]]
    return ss.active if ss.active in ids else ids[-1]


def kp_insert(text):
    rid = target_row()
    ss[f"expr_{rid}"] = ss.get(f"expr_{rid}", "") + mw.key_text(text)


def kp_action(name):
    if name == "new":
        add_row("")
        return
    rid = target_row()
    if name == "bs":
        ss[f"expr_{rid}"] = ss.get(f"expr_{rid}", "")[:-1]
    elif name == "clear":
        ss[f"expr_{rid}"] = ""


def kp_flip(step):
    names = [p["name"] for p in mw.keypages(ss.kp_base, ["t"])]
    cur = names.index(ss.kp_page) if ss.get("kp_page") in names else 0
    ss.kp_page = names[(cur + step) % len(names)]


def kp_reset_file():
    ss.kp_base, ss.kp_msg = mw.load_keypages()
    ss.kp_custom = False
    ss._kp_first = True


def view_shift(side, fwd=0):
    """移動ボタン。2D: 左右。3D: 今向いている方向を基準にした 左右 / 奥・手前"""
    if mw.SPACES[ss.space]["dim"] == 3:
        new = mw.view_shift_3d([float(ss[f"lim{i}"]) for i in range(6)], float(ss.get("azim", -60.0)), side, fwd)
    else:
        lo, hi = float(ss["lim0"]), float(ss["lim1"])
        d = (hi - lo) * 0.25 * side
        new = [lo + d, hi + d]
    for i in range(len(new)):
        ss[f"lim{i}"] = float(new[i]) if i < 4 else ss[f"lim{i}"]


def view_rot_reset():
    ss.elev, ss.azim = 25.0, -60.0
    reset_lim()


def handle_gesture():
    """指の操作 (ドラッグ・ピンチ・回転) を軸範囲・視点に反映する。グラフを描く前に呼ぶ"""
    g = ss.get("gesture")
    if not isinstance(g, dict) or g.get("id") == ss.get("_gid"):
        return
    ss._gid = g.get("id")
    ss._draft = (g.get("type") == "rot")      # 回転の最中だけ軽い画像で描く (止まると settle が来て高画質に戻る)
    S = mw.SPACES[ss.space]
    n = 2 if S["dim"] == 2 or S.get("tz") else 3
    try:
        if g["type"] == "rot" and S["dim"] == 3:
            ss.azim = (float(ss.get("azim", -60.0)) - float(g["dx"]) * 240.0 + 180.0) % 360.0 - 180.0
            ss.elev = max(-90.0, min(90.0, float(ss.get("elev", 25.0)) - float(g["dy"]) * 240.0))
            return
        if g["type"] == "view" and S["dim"] == 2:
            base = ss.get("_eff") or [float(ss[f"lim{i}"]) for i in range(6)]
            new = mw.view_transform_2d(base, g["s"], float(g["tx"]), float(g["ty"]))
        elif g["type"] in ("zoom", "view"):
            new = mw.zoom_lims([float(ss[f"lim{i}"]) for i in range(6)], g["s"], n)
        else:                                   # settle など
            return
        for i in range(2 * n):
            ss[f"lim{i}"] = float(new[i])
    except Exception:
        pass


# ======================= 左: 式の入力 =======================
def row_editor(previews):
    for idx, r in enumerate(ss.rows):
        rid = r["id"]
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1, 7, 1.2, 1.2], vertical_alignment="center")
            c1.checkbox("表示", key=f"on_{rid}", label_visibility="collapsed")
            c2.text_input(f"式 {idx + 1}", key=f"expr_{rid}", label_visibility="collapsed", placeholder="式を入力")
            c3.button("●" if ss.active == rid else "○", key=f"sel_{rid}", on_click=set_active, args=(rid,),
                      help="キーパッドの入力先にする (●が入力先)")
            c4.button("✕", key=f"rm_{rid}", on_click=rm_row, args=(rid,), help="この式を削除")
            tex, err = previews[idx] if idx < len(previews) else (None, None)
            if err:
                st.caption(f":red[{err}]")
            elif tex:
                st.latex(tex)
            d1, d2, d3 = st.columns([2, 3, 4], vertical_alignment="center")
            with d1.popover("色"):
                st.color_picker("色", key=f"col_{rid}")
            d2.number_input("太さ", 0.5, 12.0, step=0.5, format="%.2f", key=f"lw_{rid}")
            d3.markdown(f"<span style='color:{ss.get(f'col_{rid}', '#999')};font-size:26px'>━━</span>", unsafe_allow_html=True)
    st.button("＋ 式を追加", key="add_row", on_click=kp_action, args=("new",))


def keypad(names):
    pages = mw.keypages(ss.kp_base, names)
    pnames = [p["name"] for p in pages]
    if ss.pop("_kp_first", False) or ss.get("kp_page") not in pnames:
        ss.kp_page = pnames[0]                      # (部品を作る前なので書き換えてよい)
    with st.container(border=True, key="keypad"):
        with st.container(key="kpnav"):
            n1, n2, n3, n4 = st.columns([1, 5, 1, 1], vertical_alignment="center")
            n1.button("◀", key="kp_prev", on_click=kp_flip, args=(-1,), help="前のページ")
            n2.selectbox("キーパッドのページ", pnames, key="kp_page", label_visibility="collapsed")
            n3.button("▶", key="kp_next", on_click=kp_flip, args=(1,), help="次のページ")
            with n4.popover("⚙"):
                st.caption(("セッション中に読み込んだ設定を使用中" if ss.kp_custom else ss.kp_msg))
                up = st.file_uploader("キー設定JSON (mathplot_studio_keys.json) を読み込む", type=["json"], key="kp_upload")
                if up is not None and ss.get("_kp_sig") != (up.name, up.size):
                    ss._kp_sig = (up.name, up.size)
                    try:
                        ss.kp_base = mw.parse_keypages(json.loads(up.getvalue().decode("utf-8")))
                        ss.kp_custom = True
                        ss._kp_first = True
                        st.rerun()
                    except Exception as ex:
                        st.error(f"読み込めませんでした: {ex}")
                st.button("ファイルの設定に戻す", key="kp_reset", on_click=kp_reset_file)
                st.caption("リポジトリに mathplot_studio_keys.json を置くと、全員の初期キーパッドになります。")
        page = next(p for p in pages if p["name"] == ss.kp_page)
        st.caption(f"ページ {pnames.index(page['name']) + 1} / {len(pnames)}:  **{page['name']}**")
        keys = page["keys"]
        for r0 in range(0, len(keys), 7):
            cols = st.columns(7)
            for j, (lab, txt) in enumerate(keys[r0:r0 + 7]):
                cols[j].button(lab, key=f"kp_{pnames.index(page['name'])}_{r0 + j}", on_click=kp_insert, args=(txt,), help=txt)
        cols = st.columns(3)
        cols[0].button("⌫", key="kp_bs", on_click=kp_action, args=("bs",), help="1字消す")
        cols[1].button("全消去", key="kp_clear", on_click=kp_action, args=("clear",))
        cols[2].button("＋式", key="kp_new", on_click=kp_action, args=("new",))
    st.caption("○/● … キーパッドで入力する式を選びます。閉じ括弧は自動で補います。")


def left_panel():
    names, previews = mw.analyze(get_state())
    c1, c2 = st.columns([1, 5], vertical_alignment="center")
    c1.markdown("空間:")
    c2.selectbox("空間", mw.SPACE_NAMES, key="space_sel", on_change=on_space, label_visibility="collapsed")
    row_editor(previews)
    st.text_input("定数", key="consts", placeholder="定数 例: n=2, a=pi/3", label_visibility="collapsed")
    keypad(names)


# ======================= 右: グラフ =======================
def clamp_params():
    """スライダーの値を範囲内・正しい型に直す。再生中なら1コマ進める"""
    s = get_state()
    for p, q in zip(ss.params, s["params"]):
        i = p["id"]
        lo, hi = mw.param_range(q)
        v = float(ss.get(f"pv_{i}", lo))
        if q["kind"] == "int":
            a, b = math.ceil(lo), math.floor(hi)
            b = a + 1 if b <= a else b
            if ss.get(f"play_{i}"):
                v = v + 1 if v + 1 <= b else a
            ss[f"pv_{i}"] = int(round(min(max(v, a), b)))
        else:
            if ss.get(f"play_{i}"):
                v += (hi - lo) / 80
                if v > hi:
                    v = lo
            ss[f"pv_{i}"] = float(min(max(v, lo), hi))


def toolbar(dim3):
    with st.container(key="toolbar"):
        c = st.columns([1, 1, 1, 1, 1, 3, 3, 2] if dim3 else [1, 1, 1, 3, 3, 2], vertical_alignment="center")
        c[0].button("⟲", key="vb_reset", on_click=view_rot_reset, help="表示を元に戻す")
        if dim3:                                    # 3D: 今向いている方向を基準に移動
            c[1].button("◀", key="vb_left", on_click=view_shift, args=(-1, 0), help="左へ移動 (今向いている方向の左)")
            c[2].button("▶", key="vb_right", on_click=view_shift, args=(1, 0), help="右へ移動 (今向いている方向の右)")
            c[3].button("▲", key="vb_fwd", on_click=view_shift, args=(0, 1), help="奥へ移動")
            c[4].button("▼", key="vb_back", on_click=view_shift, args=(0, -1), help="手前へ移動")
        else:
            c[1].button("◀", key="vb_left", on_click=view_shift, args=(-1,), help="左へ移動")
            c[2].button("▶", key="vb_right", on_click=view_shift, args=(1,), help="右へ移動")
        c[-3].toggle("指で操作", key="touch", help="オフにすると、グラフの上でもページをスクロールできます")
        c[-2].download_button("式を保存", data=json.dumps(get_state_for_save(), ensure_ascii=False, indent=1),
                              file_name="expressions.json", mime="application/json", help="式を保存(JSON)")
        with c[-1].popover("読込"):
            up = st.file_uploader("保存したJSONを読み込む", type=["json"], key="uploader")
            if up is not None and ss.get("_loaded_sig") != (up.name, up.size):
                ss._loaded_sig = (up.name, up.size)
                try:
                    d = json.loads(up.getvalue().decode("utf-8"))
                    ss._pending = d["panels"][0] if "panels" in d else d
                    st.rerun()
                except Exception as ex:
                    st.error(f"読み込めませんでした: {ex}")


def param_bars(state):
    ok = mw.param_name_ok(state["params"])
    for p, q, good in zip(ss.params, state["params"], ok):
        i = p["id"]
        lo, hi = mw.param_range(q)
        with st.container(key=f"pbar_{i}"):
            c = st.columns([1.2, 1.7, 1.3, 5, 1.3, 1.5, 1.2, 0.9], vertical_alignment="center")
            c[0].text_input("名前", key=f"pname_{i}", disabled=p["fixed"], label_visibility="collapsed", on_change=mark_full)
            c[1].selectbox("種類", ["実数", "整数"], key=f"pkind_{i}", label_visibility="collapsed")
            c[2].text_input("最小", key=f"plo_{i}", label_visibility="collapsed")
            if q["kind"] == "int":
                a, b = math.ceil(lo), math.floor(hi)
                b = a + 1 if b <= a else b
                c[3].slider(q["name"] or "?", a, b, step=1, key=f"pv_{i}", label_visibility="collapsed")
            else:
                c[3].slider(q["name"] or "?", float(lo), float(hi), step=float((hi - lo) / 200), format="%.4g",
                            key=f"pv_{i}", label_visibility="collapsed")
            c[4].text_input("最大", key=f"phi_{i}", label_visibility="collapsed")
            c[5].markdown(f"= {float(ss.get(f'pv_{i}', 0)):.4g}")
            c[6].toggle("▶", key=f"play_{i}", help="自動で動かす")
            if not p["fixed"]:
                c[7].button("✕", key=f"prm_{i}", on_click=rm_param, args=(i,), help="このパラメータを削除")
            if not good:
                st.caption(":red[この名前は使えません (重複・予約語・数字始まりなど)]")
    st.button("＋ パラメータ", key="add_param", on_click=add_param_cb)


def build_graph(play):
    @st.fragment(run_every=0.5 if play else None)
    def graph():
        S = mw.SPACES[ss.space]
        dim3 = S["dim"] == 3
        handle_gesture()                      # 指の操作を先に反映してから描く
        clamp_params()
        state = get_state()
        toolbar(dim3)
        # ---- グラフ ----
        png = None
        draft = bool(ss.pop("_draft", False)) or play     # 回転中・再生中は軽い画像
        try:
            img, errs, eff, kind, ikey = mw.render(state, 8.0, 8.0 * float(ss.get("hfac", 0.9)), 130, draft)
            png = None if kind == "jpeg" else img
            ss._eff = eff
            # rev は内容が同じなら同じ値 → 画像を送り直さずに済む
            graph_view(f"data:image/{kind};base64," + base64.b64encode(img).decode(), "3d" if dim3 else "2d",
                       ikey, bool(ss.get("touch", True)), key="gesture")
            for i, e in enumerate(errs):
                if e:
                    st.caption(f":red[式 {i + 1}: {e}]")
        except Exception as ex:
            st.error(f"描画できませんでした: {ex}")
        param_bars(state)
        # ---- 軸・スケール ----
        with st.expander("軸・スケール"):
            n = 2 if S["dim"] == 2 or S.get("tz") else 3
            for i in range(n):
                c = st.columns([2, 1, 2], vertical_alignment="center")
                c[0].number_input(f"{S['labels'][i]} 最小", key=f"lim{2 * i}", step=1.0, format="%.3f")
                c[1].markdown("〜")
                c[2].number_input(f"{S['labels'][i]} 最大", key=f"lim{2 * i + 1}", step=1.0, format="%.3f")
            st.checkbox("縦横(各軸)を等倍にする", key="equal")
            if S.get("tz"):
                opts = ["t"] + [p["name"].strip() for p in state["params"] if not p["fixed"] and p["name"].strip()]
                if ss.get("axp") not in opts:
                    ss.axp = "t"
                st.selectbox("第3軸にするパラメータ", opts, key="axp")
            if dim3:
                st.slider("見る角度 (上下)", -90.0, 90.0, step=1.0, format="%.0f", key="elev")
                st.slider("見る角度 (回転)", -180.0, 180.0, step=1.0, format="%.0f", key="azim")
            st.slider("グラフの縦の長さ", 0.5, 1.3, step=0.05, key="hfac")
            if png is not None:
                st.download_button("グラフを画像(PNG)で保存", data=png, file_name="graph.png", mime="image/png")
        # 再生の ON/OFF が切り替わった / 式の入力欄にも関わる変更があったときは、全体を再実行
        now = any(ss.get(f"play_{p['id']}") for p in ss.params)
        need = ss.pop("_need_full", False)
        if now != ss.get("_decor_play") or need:
            st.rerun()
    return graph


# ======================= メイン =======================
init()
if ss.get("_pending") is not None:          # JSON読込 (部品が作られる前に状態を差し替える)
    apply_dict(ss.pop("_pending"))

if ICON is not None:
    h0, h1, h2 = st.columns([1, 6, 4], vertical_alignment="center")
    h0.image(ICON, width=44)
else:
    h1, h2 = st.columns([7, 4], vertical_alignment="center")
h1.markdown(f"### {APP_NAME}")
h2.radio("レイアウト", ["横並び", "縦並び"], key="layout", horizontal=True, label_visibility="collapsed",
         help="スマホは縦並び、iPad・PCは横並びが初期設定です")

play = any(ss.get(f"play_{p['id']}") for p in ss.params)
ss._decor_play = play
graph = build_graph(play)

if ss.layout == "横並び":
    col_l, col_r = st.columns([4, 6])
else:
    col_r = st.container()      # 縦並び: グラフを上、入力を下に置く (コードの実行順は 入力 → グラフ)
    col_l = st.container()
with col_l:
    left_panel()
with col_r:
    graph()
