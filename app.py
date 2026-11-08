import streamlit as st

st.set_page_config(
    page_title="MathPlot Studio",
    layout="wide"
)

st.title("MathPlot Studio")

st.write("Web版の動作確認")

expression = st.text_input(
    "数式を入力してください",
    value="sin(x)"
)

st.write("入力された式：", expression)