import os
os.environ["MPS_NOGUI"] = "1"

import json
import io
import re
import traceback
from types import SimpleNamespace

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy as sp

import mathplot_core as core

st.set_page_config(page_title="MathPlot Studio", page_icon="📈", layout="wide")
st.title("MathPlot Studio")
st.caption("Web版 — 元アプリの数式エンジン・描画関数を再利用")

SPACES = core.SPACES
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
           "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def new_panel(space="xy平面"):
    s = SPACES[space]
    lim = [-5.0, 5.0] * 3
    return {
        "space": space,
        "lim": lim,
        "equal": bool(s.get("equal", True)),
        "consts": "",
        "axp": "t",
        "params": [{
            "name": "t",
            "kind": "real",
            "lo": 0.0,
            "hi": 6.28,
            "val": 0.0,
            "fixed": True
        }],
        "rows": [{
            "expr": "",
            "color": PALETTE[0],
            "on": True,
            "lw": 2.0
        }],
        "view": [25.0, -60.0],
    }


def normalize_panel(data):
    """元デスクトップ版の to_dict() 形式をできるだけ保って読み込む。"""
    d = new_panel(
        data.get("space", "xy平面")
        if data.get("space") in SPACES
        else "xy平面"
    )

    d.update({
        k: data[k]
        for k in (
            "space", "lim", "equal",
            "consts", "axp", "params", "rows"
        )
        if k in data
    })

    if "lim" not in d:
        r = float(data.get("R", 5.0))
        d["lim"] = [-r, r] * 3

    d["params"] = d.get("params") or [{
        "name": "t",
        "kind": "real",
        "lo": data.get("tmin", 0),
        "hi": data.get("tmax", 6.28),
        "val": 0,
        "fixed": True
    }]

    d["rows"] = d.get("rows") or [{
        "expr": "",
        "color": PALETTE[0],
        "on": True,
        "lw": 2.0
    }]

    d.setdefault("view", [25.0, -60.0])
    return d


def init_state():
    if "mp_tabs" not in st.session_state:
        st.session_state.mp_tabs = [{
            "name": "グラフ 1",
            "panels": [new_panel()]
        }]
        st.session_state.mp_active = 0
        st.session_state.mp_panel = 0


init_state()


# ---- タブ / パネル管理 ----

head = st.columns([1, 1, 1, 2])

if head[0].button("＋ 新しいタブ", use_container_width=True):
    st.session_state.mp_tabs.append({
        "name": f"グラフ {len(st.session_state.mp_tabs)+1}",
        "panels": [new_panel()]
    })
    st.session_state.mp_active = len(st.session_state.mp_tabs)-1
    st.session_state.mp_panel = 0
    st.rerun()

if head[1].button("＋ パネル", use_container_width=True):
    tab = st.session_state.mp_tabs[st.session_state.mp_active]
    tab["panels"].append(new_panel(tab["panels"][0]["space"]))
    st.session_state.mp_panel = len(tab["panels"])-1
    st.rerun()

if head[2].button(
    "パネルを削除",
    use_container_width=True,
    disabled=len(st.session_state.mp_tabs[
        st.session_state.mp_active
    ]["panels"]) <= 1
):
    tab = st.session_state.mp_tabs[st.session_state.mp_active]
    tab["panels"].pop(st.session_state.mp_panel)
    st.session_state.mp_panel = max(
        0,
        min(st.session_state.mp_panel, len(tab["panels"])-1)
    )
    st.rerun()


names = [t["name"] for t in st.session_state.mp_tabs]

st.session_state["tab_selector"] = min(
    st.session_state.mp_active,
    len(names)-1
)

st.session_state.mp_active = min(
    st.session_state.mp_active,
    len(names)-1
)

st.session_state.mp_active = st.selectbox(
    "タブ",
    range(len(names)),
    format_func=lambda i: names[i],
    index=st.session_state.mp_active,
    key="tab_selector"
)

tab = st.session_state.mp_tabs[st.session_state.mp_active]

st.session_state["panel_selector"] = min(
    st.session_state.mp_panel,
    len(tab["panels"])-1
)

st.session_state.mp_panel = min(
    st.session_state.mp_panel,
    len(tab["panels"])-1
)

st.session_state.mp_panel = st.selectbox(
    "このタブ内のパネル",
    range(len(tab["panels"])),
    format_func=lambda i: f"パネル {i+1}",
    index=st.session_state.mp_panel,
    key="panel_selector"
)

panel = tab["panels"][st.session_state.mp_panel]
PFX = f"{st.session_state.mp_active}_{st.session_state.mp_panel}"


# ---- JSON保存・読込 ----

with st.expander(
    "保存・読込（元アプリのJSON形式に対応）",
    expanded=False
):
    payload = {"panels": tab["panels"]}

    st.download_button(
        "このタブをJSON保存",
        json.dumps(payload, ensure_ascii=False, indent=2),
        file_name="mathplot_graph.json",
        mime="application/json"
    )

    uploaded = st.file_uploader(
        "元アプリまたはWeb版のJSONを読み込む",
        type=["json"],
        key="json_upload"
    )

    if uploaded is not None and st.button(
        "JSONを読み込む",
        key="load_json"
    ):
        try:
            incoming = json.load(uploaded)

            if isinstance(incoming, dict) and "panels" in incoming:
                tab["panels"] = [
                    normalize_panel(x)
                    for x in incoming["panels"]
                ]
            else:
                tab["panels"] = [normalize_panel(incoming)]

            st.session_state.mp_panel = 0
            st.success("JSONを読み込みました")
            st.rerun()

        except Exception as exc:
            st.error(f"JSONを読み込めませんでした: {exc}")


# ---- グラフ設定 ----

space_names = list(SPACES)

with st.container(border=True):
    st.subheader("グラフ設定")

    c1, c2, c3 = st.columns([2, 1, 1])

    current_space = panel.get("space", "xy平面")

    selected_space = c1.selectbox(
        "空間",
        space_names,
        index=(
            space_names.index(current_space)
            if current_space in space_names else 0
        ),
        key=f"space_{st.session_state.mp_active}_{st.session_state.mp_panel}"
    )

    if selected_space != current_space:
        panel["space"] = selected_space
        panel["lim"] = [-5.0, 5.0] * 3
        panel["equal"] = SPACES[selected_space].get("equal", True)
        st.rerun()

    S = SPACES[selected_space]
    dim = S["dim"]
    tz = S.get("tz", False)

    c2.checkbox(
        "各軸の縮尺を等倍",
        value=panel.get("equal", True),
        key=f"equal_widget_{PFX}"
    )

    panel["equal"] = st.session_state[f"equal_widget_{PFX}"]

    c3.number_input(
        "描画線幅の初期値",
        min_value=0.2,
        max_value=8.0,
        value=2.0,
        step=0.2,
        key=f"default_lw_{PFX}"
    )

    st.markdown("**表示範囲**")

    lim = list(panel.get("lim", [-5, 5] * 3))
    axis_labels = list(S["labels"])

    if tz:
        axis_labels[2] = panel.get("axp", "t")

    range_cols = st.columns(dim)

    for i in range(dim):
        with range_cols[i]:
            a, b = st.columns(2)

            lim[2*i] = a.number_input(
                f"{axis_labels[i]} 最小",
                value=float(lim[2*i]),
                key=f"lim_{i}_lo_{PFX}"
            )

            lim[2*i+1] = b.number_input(
                f"{axis_labels[i]} 最大",
                value=float(lim[2*i+1]),
                key=f"lim_{i}_hi_{PFX}"
            )

            if lim[2*i] >= lim[2*i+1]:
                st.error(
                    f"{axis_labels[i]} の最小値は最大値より小さくしてください"
                )

    panel["lim"] = lim

    if dim == 3:
        v1, v2 = st.columns(2)

        panel["view"][0] = v1.slider(
            "3D仰角",
            -90, 90,
            int(panel["view"][0]),
            key=f"view_elev_{PFX}"
        )

        panel["view"][1] = v2.slider(
            "3D方位角",
            -180, 180,
            int(panel["view"][1]),
            key=f"view_azim_{PFX}"
        )

    panel["consts"] = st.text_input(
        "定数（例: a=2, b=pi/3）",
        value=panel.get("consts", ""),
        key=f"consts_input_{PFX}"
    )


# ---- パラメータ ----

st.subheader("パラメータ")

params = panel.get("params", [])

if not any(p.get("name") == "t" for p in params):
    params.insert(0, {
        "name": "t",
        "kind": "real",
        "lo": 0.0,
        "hi": 6.28,
        "val": 0.0,
        "fixed": True
    })

for pi, p in enumerate(params):
    with st.container(border=True):
        a, b, c, d = st.columns([1.1, 1, 1, 1.4])

        p["name"] = a.text_input(
            "名前",
            value=p.get("name", "t"),
            key=f"param_name_{pi}_{PFX}"
        )

        p["kind"] = b.selectbox(
            "型",
            ["real", "int"],
            index=1 if p.get("kind") == "int" else 0,
            key=f"param_kind_{pi}_{PFX}"
        )

        p["lo"] = c.number_input(
            "最小",
            value=float(p.get("lo", 0)),
            key=f"param_lo_{pi}_{PFX}"
        )

        p["hi"] = d.number_input(
            "最大",
            value=float(p.get("hi", 6.28)),
            key=f"param_hi_{pi}_{PFX}"
        )

        if p["lo"] > p["hi"]:
            st.warning(
                f"{p['name']}: 最小値と最大値を確認してください"
            )

        if p["kind"] == "int":
            vlo = int(np.ceil(p["lo"]))
            vhi = int(np.floor(p["hi"]))

            if vlo > vhi:
                vlo, vhi = int(p["lo"]), int(p["lo"])

            current = min(
                max(int(p.get("val", vlo)), vlo),
                vhi
            )

            p["val"] = st.slider(
                f"{p['name']} の値",
                vlo,
                vhi,
                current,
                key=f"param_val_{pi}_{PFX}"
            )

        else:
            vlo, vhi = float(p["lo"]), float(p["hi"])

            current = min(
                max(float(p.get("val", vlo)), vlo),
                vhi
            )

            p["val"] = st.slider(
                f"{p['name']} の値",
                vlo,
                max(vlo+1e-9, vhi),
                current,
                key=f"param_val_{pi}_{PFX}"
            )

        if p["name"] != "t" and st.button(
            f"{p['name']} を削除",
            key=f"param_del_{pi}_{PFX}"
        ):
            params.pop(pi)
            st.rerun()

if st.button("＋ パラメータを追加"):
    used = {p.get("name") for p in params}

    name = next(
        (x for x in "abcdfghpquvw" if x not in used),
        "p"
    )

    params.append({
        "name": name,
        "kind": "real",
        "lo": -5.0,
        "hi": 5.0,
        "val": 0.0,
        "fixed": False
    })

    st.rerun()

panel["params"] = params

if tz:
    eligible = [p.get("name", "t") for p in params]
    default_axp = panel.get("axp", "t")

    panel["axp"] = st.selectbox(
        "第3軸にするパラメータ",
        eligible,
        index=(
            eligible.index(default_axp)
            if default_axp in eligible else 0
        ),
        key=f"axis_parameter_{PFX}"
    )


# ---- 数式・描画項目 ----

st.subheader("数式・描画項目")

rows = panel.get("rows", [])

for ri, row in enumerate(rows):
    with st.container(border=True):
        a, b, c, d = st.columns([5, 0.8, 1, 0.8])

        row["expr"] = a.text_input(
            f"式 {ri+1}",
            value=row.get("expr", ""),
            key=f"expr_{ri}_{PFX}",
            placeholder="例: y=sin(x) / (x(s),y(s)) / vec(1,2)"
        )

        row["on"] = b.checkbox(
            "表示",
            value=bool(row.get("on", True)),
            key=f"enabled_{ri}_{PFX}"
        )

        row["color"] = c.color_picker(
            "色",
            value=row.get(
                "color",
                PALETTE[ri % len(PALETTE)]
            ),
            key=f"color_{ri}_{PFX}"
        )

        row["lw"] = d.number_input(
            "線幅",
            min_value=0.2,
            max_value=8.0,
            value=float(row.get("lw", 2.0)),
            step=0.2,
            key=f"lw_{ri}_{PFX}"
        )

        if st.button(
            f"式 {ri+1} を削除",
            key=f"row_del_{ri}_{PFX}"
        ):
            rows.pop(ri)
            st.rerun()

if st.button("＋ 式を追加"):
    rows.append({
        "expr": "",
        "color": PALETTE[len(rows) % len(PALETTE)],
        "on": True,
        "lw": 2.0
    })
    st.rerun()

panel["rows"] = rows


# ---- 数式キーパッド ----

with st.expander("数式キーパッド（入力用テンプレート）", expanded=False):

    key_groups = {
        "基本": [
            "sin(|)", "cos(|)", "tan(|)", "sqrt(|)",
            "abs(|)", "pi", "^2", "^(|)", "exp(|)",
            "log(|)", "vec(|)", "field(|)", "basis(|)"
        ],

        "微積・和": [
            "diff(|,x)",
            "integral(|,x,0,1)",
            "sum(|,k,1,10)",
            "prod(|,k,1,10)"
        ],

        "ベクトル・行列": [
            "dot(|,)",
            "cross(|,)",
            "norm(|)",
            "det(|)",
            "mat2(|,,,)"
        ],

        "特殊関数": [
            "gamma(|)",
            "zeta(|)",
            "besselj(|,)",
            "airyai(|)",
            "lambertw(|)",
            "polylog(|,)"
        ],
    }

    for group, keys in key_groups.items():
        st.markdown(f"**{group}**")
        st.code("    ".join(keys), language=None)


# ---- グラフ描画 ----

st.divider()
st.subheader("グラフ")

errors = []

try:
    # 元アプリと同じ数式環境を設定する
    valid_params = []
    seen = {"t"}

    core.PVAL.clear()

    for p in params:
        name = str(p.get("name", "")).strip()

        if not name or name in seen or core.param_reserved(name):
            if name != "t":
                errors.append(f"無効なパラメータ名: {name}")
            continue

        seen.add(name)
        core.PVAL[name] = float(p.get("val", 0))

        if name != "t":
            valid_params.append(name)

    trow = next(
        (p for p in params if p.get("name") == "t"),
        params[0]
    )

    core.PVAL["t"] = float(trow.get("val", 0))

    core.set_params(valid_params)

    core.AX[0] = panel.get("axp", "t") if tz else "t"

    core.CONSTS.clear()

    for part in re.split(r"[,;]", panel.get("consts", "")):
        if "=" in part:
            k, v = part.split("=", 1)

            try:
                core.CONSTS[k.strip()] = core.parse_expr(
                    v.strip().replace("π", "pi"),
                    local_dict=core.LOC,
                    transformations=core.TF
                )
            except Exception as ex:
                errors.append(f"定数 {k.strip()}: {ex}")

    key = S["key"]

    def_errors = core.build_defs(
        [r.get("expr", "") for r in rows],
        key
    )

    R = max(
        abs(float(v))
        for i in range(2 if tz or dim == 2 else 3)
        for v in lim[2*i:2*i+2]
    )

    axis_param = next(
        (p for p in params if p.get("name") == core.AX[0]),
        trow
    )

    tr = (
        float(axis_param.get("lo", 0)),
        float(axis_param.get("hi", 6.28))
    )

    ctx = SimpleNamespace(
        R=max(R, 1e-6),
        t=float(axis_param.get("val", 0)),
        tr=tr,
        cond="",
        lw=2.0,
        maps=[]
    )

    if key == "cx":
        for row in rows:
            m = re.match(
                core.MAP_RE,
                core.split_cond(row.get("expr", ""))[0]
            )

            if m and row.get("on", True):
                try:
                    ctx.maps.append(
                        core.compile_expr(m.group(1))[2]
                    )
                except Exception as ex:
                    errors.append(f"写像式: {ex}")

    fig = plt.figure(
        figsize=(10, 7 if dim == 2 else 8),
        dpi=110
    )

    ax = fig.add_axes(
        [0.055, 0.065, 0.94, 0.93]
        if dim == 2
        else [0.02, 0.02, 0.96, 0.96],
        projection="3d" if dim == 3 else None
    )

    for i, row in enumerate(rows):
        expr = row.get("expr", "").strip()

        if def_errors.get(i):
            errors.append(
                f"式 {i+1} 定義エラー: {def_errors[i]}"
            )
            continue

        if not row.get("on", True) or not expr:
            continue

        try:
            ctx.lw = float(row.get("lw", 2.0))

            with matplotlib.rc_context({
                "lines.linewidth": ctx.lw
            }):
                core.draw_item(
                    ax,
                    expr,
                    S,
                    ctx,
                    row.get(
                        "color",
                        PALETTE[i % len(PALETTE)]
                    )
                )

        except Exception as ex:
            errors.append(f"式 {i+1}: {ex}")

    if S.get("polar"):
        th = np.linspace(0, 2*np.pi, 200)

        for k in range(1, 40):
            ax.plot(
                k*R/5*np.cos(th),
                k*R/5*np.sin(th),
                color="gray",
                lw=0.5,
                alpha=0.4
            )

        for a in np.arange(0, 2*np.pi, np.pi/6):
            ax.plot(
                [0, 20*R*np.cos(a)],
                [0, 20*R*np.sin(a)],
                color="gray",
                lw=0.5,
                alpha=0.4
            )

    labels = list(S["labels"])

    if tz:
        labels[2] = core.AX[0]

    if dim == 2:
        if panel.get("equal", True) and not S.get("polar"):
            xspan = lim[1]-lim[0]
            yspan = lim[3]-lim[2]
            ratio = xspan / max(yspan, 1e-9)

            if ratio > 1:
                lim[2] = (lim[2]+lim[3])/2 - xspan/2
                lim[3] = lim[2]+xspan
            else:
                lim[0] = (lim[0]+lim[1])/2 - yspan/2
                lim[1] = lim[0]+yspan

        core.axes2d(ax, lim, labels, 8)

        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])

    else:
        ax.set_xlim3d(lim[0], lim[1])
        ax.set_ylim3d(lim[2], lim[3])
        ax.set_zlim3d(lim[4], lim[5])

        core.axes3d(ax, lim, labels)

        ax.view_init(
            elev=float(panel.get("view", [25, -60])[0]),
            azim=float(panel.get("view", [25, -60])[1])
        )

        if panel.get("equal", True):
            ax.set_box_aspect([
                lim[1]-lim[0],
                lim[3]-lim[2],
                lim[5]-lim[4]
            ])

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

except Exception as exc:
    st.error(f"グラフを描画できませんでした: {exc}")
    st.code(traceback.format_exc())


if errors:
    with st.expander(
        f"式・設定の確認事項 ({len(errors)}件)",
        expanded=True
    ):
        for err in errors:
            st.warning(err)


st.caption(
    "注: 数式パーサ、SymPy/NumPy評価、元アプリの描画関数を直接利用しています。"
    "Web画面・タッチ操作・複数パネルの配置はWeb向けに置換中です。"
)
