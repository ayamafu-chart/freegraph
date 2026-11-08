"""MathPlot Studio (Web版) - iPad の Chrome など、タッチ操作のブラウザ向け画面

構成:  app.py (この画面)  →  mathplot_web.py (橋渡し)  →  mathplot_core.py (数式の解析・描画)
"""
import json
import math

import streamlit as st

import mathplot_web as mw

st.set_page_config(page_title="MathPlot Studio", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

# ======================= タッチ操作向けの見た目 =======================
st.markdown("""
<style>
/* 余白を詰めて画面を広く使う */
.block-container {padding: 2.6rem 0.9rem 3rem 0.9rem; max-width: 100%;}
/* iPad(Safari系)は文字サイズが16px未満の入力欄をタップすると勝手に拡大するので、16px以上にする */
input, textarea, select, [data-baseweb="select"] * {font-size: 17px !important;}
/* ボタンは指で押しやすい大きさに。ダブルタップでの拡大も防ぐ */
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button {
    min-height: 46px; min-width: 46px; touch-action: manipulation;}
/* スライダーのつまみを大きく */
[data-testid="stSlider"] [role="slider"] {width: 30px !important; height: 30px !important;}
[data-testid="stSlider"] {padding-top: 0.2rem; padding-bottom: 0.4rem;}
/* キーパッド・操作ボタンは横一列のまま折り返さない (画面が狭くても崩れない) */
.st-key-keypad [data-testid="stHorizontalBlock"], .st-key-viewbtns [data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important; gap: 0.25rem !important;}
.st-key-keypad [data-testid="stColumn"], .st-key-viewbtns [data-testid="stColumn"],
.st-key-keypad [data-testid="column"], .st-key-viewbtns [data-testid="column"] {
    min-width: 0 !important; flex: 1 1 0 !important; width: auto !important;}
.st-key-keypad button, .st-key-viewbtns button {width: 100%; padding: 0 !important; font-size: 17px;}
/* グラフ画像は欄の幅いっぱいに */
[data-testid="stImage"] img {width: 100% !important; height: auto !important;}
h3 {margin: 0 0 0.3rem 0; padding: 0;}
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
    add_row(mw.EXAMPLES[ss.space])
    for i in range(6):
        ss[f"lim{i}"] = -5.0 if i % 2 == 0 else 5.0
    ss.equal, ss.consts, ss.axp = True, "", "t"
    ss.elev, ss.azim, ss.hfac = 25, -60, 0.9
    ss.layout = "横並び"


def reset_lim():
    for i in range(6):
        ss[f"lim{i}"] = -5.0 if i % 2 == 0 else 5.0


def on_space():
    ss.space = ss.space_sel
    ss.rows = []
    reset_lim()
    add_row(mw.EXAMPLES[ss.space])
    ss.equal = True


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
                elev=float(ss.get("elev", 25)), azim=float(ss.get("azim", -60)))


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


def rm_param(pid): ss.params = [p for p in ss.params if p["id"] != pid]


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


def view_move(kind):
    """ズーム・移動ボタン (マウスのホイール/ドラッグの代わり)"""
    S = mw.SPACES[ss.space]
    n = 2 if S["dim"] == 2 or S.get("tz") else 3
    for i in range(n):
        lo, hi = float(ss[f"lim{2 * i}"]), float(ss[f"lim{2 * i + 1}"])
        c, w = (lo + hi) / 2, hi - lo
        if kind in ("in", "out"):
            f = 0.8 if kind == "in" else 1.25
            lo, hi = c - w * f / 2, c + w * f / 2
        else:
            d = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}[kind][i] if i < 2 else 0
            lo, hi = lo + d * w * 0.25, hi + d * w * 0.25
        ss[f"lim{2 * i}"], ss[f"lim{2 * i + 1}"] = lo, hi


def view_rot(de, da):
    ss.elev = max(-90, min(90, int(ss.get("elev", 25)) + de))
    a = int(ss.get("azim", -60)) + da
    ss.azim = (a + 180) % 360 - 180


# ======================= 画面の部品 =======================
def row_editor(names, previews):
    for idx, r in enumerate(ss.rows):
        rid = r["id"]
        c1, c2, c3, c4 = st.columns([1, 7, 1.3, 1.3], vertical_alignment="center")
        c1.checkbox("表示", key=f"on_{rid}", label_visibility="collapsed")
        c2.text_input(f"式 {idx + 1}", key=f"expr_{rid}", label_visibility="collapsed", placeholder="式を入力  例: sin(x)")
        c3.button("●" if ss.active == rid else "✎", key=f"sel_{rid}", on_click=set_active, args=(rid,),
                  help="キーパッドの入力先にする")
        c4.button("✕", key=f"rm_{rid}", on_click=rm_row, args=(rid,), help="この式を削除")
        tex, err = previews[idx] if idx < len(previews) else (None, None)
        d1, d2, d3 = st.columns([1, 7, 2.6], vertical_alignment="center")
        d1.markdown(f"<span style='color:{ss.get(f'col_{rid}', '#999')};font-size:24px'>■</span>", unsafe_allow_html=True)
        with d2:
            if err:
                st.caption(f":red[{err}]")
            elif tex:
                st.latex(tex)
        with d3.popover("色・太さ"):
            st.color_picker("色", key=f"col_{rid}")
            st.slider("太さ", 0.5, 12.0, step=0.5, key=f"lw_{rid}")
    st.button("＋ 式を追加", on_click=kp_action, args=("new",))


def keypad(names):
    pages = mw.keypages(names)
    with st.container(key="keypad"):
        st.selectbox("キーパッドのページ", [p["name"] for p in pages], key="kp_page", label_visibility="collapsed")
        page = next((p for p in pages if p["name"] == ss.kp_page), pages[0])
        keys = page["keys"]
        for r0 in range(0, len(keys), 7):
            cols = st.columns(7)
            for j, (lab, txt) in enumerate(keys[r0:r0 + 7]):
                cols[j].button(lab, key=f"kp_{page['name']}_{r0 + j}", on_click=kp_insert, args=(txt,), help=txt)
        cols = st.columns(3)
        cols[0].button("⌫ 1字消す", key="kp_bs", on_click=kp_action, args=("bs",))
        cols[1].button("全消去", key="kp_clear", on_click=kp_action, args=("clear",))
        cols[2].button("＋ 式", key="kp_new", on_click=kp_action, args=("new",))
    st.caption("🔹 ✎/● のボタンで入力先の式を選びます。括弧は閉じ忘れても自動で補います。")


def param_editor(names):
    with st.expander("パラメータの追加・設定 (スライダーで動かせる変数)"):
        for p in ss.params:
            i = p["id"]
            c = st.columns([2, 2, 2, 2, 1.2], vertical_alignment="bottom")
            c[0].text_input("名前", key=f"pname_{i}", disabled=p["fixed"])
            c[1].selectbox("種類", ["実数", "整数"], key=f"pkind_{i}")
            c[2].text_input("最小", key=f"plo_{i}")
            c[3].text_input("最大", key=f"phi_{i}")
            if not p["fixed"]:
                c[4].button("✕", key=f"prm_{i}", on_click=rm_param, args=(i,))
                if str(ss.get(f"pname_{i}", "")).strip() not in names:
                    st.caption(":red[この名前は使えません (重複・予約語・数字始まりなど)]")
        st.button("＋ パラメータを追加", on_click=lambda: add_param())
        st.caption("範囲には π や 2π、sqrt(2) などの式も書けます。")


def save_load():
    with st.expander("保存・読込 (JSON)"):
        st.download_button("式を保存 (JSON)", data=json.dumps(get_state_for_save(), ensure_ascii=False, indent=1),
                           file_name="expressions.json", mime="application/json")
        up = st.file_uploader("保存したJSONを読み込む", type=["json"], key="uploader")
        if up is not None:
            sig = (up.name, up.size)
            if ss.get("_loaded_sig") != sig:
                ss._loaded_sig = sig
                try:
                    d = json.loads(up.getvalue().decode("utf-8"))
                    ss._pending = d["panels"][0] if "panels" in d else d
                    st.rerun()
                except Exception as ex:
                    st.error(f"読み込めませんでした: {ex}")


def get_state_for_save():
    s = get_state()
    return dict(space=s["space"], lim=s["lim"], equal=s["equal"], consts=s["consts"], axp=s["axp"],
                params=[dict(name=p["name"], kind=p["kind"], lo=p["lo"], hi=p["hi"], val=p["val"], fixed=p["fixed"]) for p in s["params"]],
                rows=s["rows"])


def left_panel():
    names, previews = mw.analyze(get_state())
    row_editor(names, previews)
    st.text_input("定数 (例: n=2, a=pi/3)", key="consts", placeholder="n=2, a=pi/3")
    keypad(names)
    param_editor(names)
    save_load()


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


def build_graph(play):
    @st.fragment(run_every=0.5 if play else None)
    def graph():
        S = mw.SPACES[ss.space]
        clamp_params()
        # ---- ズーム・移動 ----
        with st.container(key="viewbtns"):
            cols = st.columns(7)
            for c, (lab, k) in zip(cols, [("＋", "in"), ("－", "out"), ("◀", "left"), ("▶", "right"), ("▲", "up"), ("▼", "down"), ("⟲", "reset")]):
                if k == "reset":
                    c.button(lab, key="vb_reset", on_click=reset_lim, help="表示範囲を元に戻す")
                else:
                    c.button(lab, key=f"vb_{k}", on_click=view_move, args=(k,), help={"in": "拡大", "out": "縮小"}.get(k, "移動"))
        state = get_state()
        # ---- グラフ ----
        try:
            png, errs, _ = mw.render(state, 8.0, 8.0 * float(ss.get("hfac", 0.9)), 130)
            st.image(png)
            for i, e in enumerate(errs):
                if e:
                    st.caption(f":red[式 {i + 1}: {e}]")
        except Exception as ex:
            png = None
            st.error(f"描画できませんでした: {ex}")
        # ---- パラメータのスライダー ----
        for p, q in zip(ss.params, state["params"]):
            i = p["id"]
            lo, hi = mw.param_range(q)
            nm = q["name"].strip() or "?"
            c1, c2 = st.columns([8, 2], vertical_alignment="bottom")
            if q["kind"] == "int":
                a, b = math.ceil(lo), math.floor(hi)
                b = a + 1 if b <= a else b
                c1.slider(nm, a, b, step=1, key=f"pv_{i}")
            else:
                c1.slider(nm, float(lo), float(hi), step=float((hi - lo) / 200), format="%.4g", key=f"pv_{i}")
            c2.toggle("再生", key=f"play_{i}")
        # ---- 軸・表示 ----
        with st.expander("軸・表示の設定"):
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
            if S["dim"] == 3:
                r = st.columns(4)
                r[0].button("↺ 左", key="rot_l", on_click=view_rot, args=(0, -15))
                r[1].button("↻ 右", key="rot_r", on_click=view_rot, args=(0, 15))
                r[2].button("↑ 上", key="rot_u", on_click=view_rot, args=(10, 0))
                r[3].button("↓ 下", key="rot_d", on_click=view_rot, args=(-10, 0))
                st.slider("見る角度 (上下)", -90, 90, key="elev")
                st.slider("見る角度 (回転)", -180, 180, key="azim")
            st.slider("グラフの縦の長さ", 0.5, 1.3, step=0.05, key="hfac")
            if png is not None and not play:
                st.download_button("グラフを画像(PNG)で保存", data=png, file_name="graph.png", mime="image/png")
        # 再生の ON/OFF が切り替わったら、自動更新の設定を作り直すために全体を再実行
        now = any(ss.get(f"play_{p['id']}") for p in ss.params)
        if now != ss.get("_decor_play"):
            st.rerun()
    return graph


# ======================= メイン =======================
init()
if ss.get("_pending") is not None:          # JSON読込 (部品が作られる前に状態を差し替える)
    apply_dict(ss.pop("_pending"))

h1, h2, h3 = st.columns([3, 4, 3], vertical_alignment="center")
h1.markdown("### 📈 MathPlot Studio")
h2.selectbox("空間", mw.SPACE_NAMES, key="space_sel", on_change=on_space, label_visibility="collapsed")
h3.radio("レイアウト", ["横並び", "縦並び"], key="layout", horizontal=True, label_visibility="collapsed",
         help="iPadを縦向きで使うときは「縦並び」(グラフが上、入力が下) が見やすいです")

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
