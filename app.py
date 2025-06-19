import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import textwrap

# ───── Streamlit 基本設定 ─────
st.set_page_config(
    page_title="講義の質疑応答まとめアプリ : AIE Proj 04", layout="centered"
)
st.title("📚 講義の質疑応答まとめアプリ : AIE Proj 04")

client = OpenAI(api_key=st.secrets["openai_api_key"])

# ───── サイドバー：設定項目 ─────
with st.sidebar:
    st.header("⚙️ 設定")
    num_clusters = st.slider("クラスタ数（KMeans）", 2, 20, 10)
    model_options = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"]
    summary_model = st.selectbox("代表質問生成モデル", model_options, index=1)
    answer_model = st.selectbox("模範回答生成モデル", model_options, index=2)

# ───── CSV アップロード（メインカラム）─────
uploaded_file = st.file_uploader(
    "📤 CSVファイルをアップロード（列: 質問, 回答）", type=["csv"]
)

# ───── メイン処理 ─────
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "質問" not in df.columns:
        st.error("'質問' 列が見つかりません。CSV を確認してください。")
        st.stop()

    st.success(f"✅ {len(df)} 件の質問を読み込みました。")

    # Embeddings
    with st.spinner("💠 OpenAI 埋め込みを取得中..."):
        embs = client.embeddings.create(
            model="text-embedding-ada-002", input=df["質問"].tolist()
        )
    vectors = np.array([e.embedding for e in embs.data])

    # KMeans
    labels = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(
        normalize(vectors)
    )
    df["クラスタ"] = labels
    st.success(f"✅ {num_clusters} クラスタに分類しました。")

    # ── 各クラスタ UI ──
    for cid in range(num_clusters):
        cdf = df[df["クラスタ"] == cid]
        questions = cdf["質問"].tolist()

        sum_key, ans_key = f"sum_{cid}", f"ans_{cid}"

        with st.expander(f"▶️ クラスタ {cid}：{len(questions)} 件の質問", expanded=True):

            # 1. 質問リスト
            st.markdown("\n".join([f"- {q}" for q in questions]))

            # 2. ボタン & プレースホルダー
            gen_sum = st.button("🧠 代表質問を生成", key=f"btn_sum_{cid}")
            sum_ph = st.container()
            gen_ans = st.button("💡 模範回答を生成", key=f"btn_ans_{cid}")
            ans_ph = st.container()

            # 代表質問生成
            if gen_sum:
                prompt = (
                    textwrap.dedent(
                        """\
                    以下の質問は講義中に受けた似た内容の質問です。
                    1 つに要約してください。

                    質問一覧:
                    """
                    )
                    + "\n".join([f"- {q}" for q in questions])
                    + "\n\n代表質問："
                )
                with st.spinner(f"{summary_model} で代表質問生成中..."):
                    res = client.chat.completions.create(
                        model=summary_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "あなたは要約が得意なアシスタントです。",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.5,
                    )
                st.session_state[sum_key] = res.choices[0].message.content.strip()
                st.session_state.pop(ans_key, None)
                st.toast("✅ 代表質問を生成しました")

            # 模範回答生成
            if gen_ans:
                if not st.session_state.get(sum_key):
                    st.warning("⚠️ 先に代表質問を生成してください。")
                else:
                    a_prompt = textwrap.dedent(
                        f"""\
                        以下の質問に対して、講義で使える模範回答を作成してください。

                        質問：{st.session_state[sum_key]}

                        回答：
                        """
                    )
                    with st.spinner(f"{answer_model} で模範回答生成中..."):
                        ares = client.chat.completions.create(
                            model=answer_model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "あなたはわかりやすい回答を作る講義アシスタントです。",
                                },
                                {"role": "user", "content": a_prompt},
                            ],
                            temperature=0.7,
                        )
                    st.session_state[ans_key] = ares.choices[0].message.content.strip()
                    st.toast("✅ 模範回答を生成しました")

            # プレースホルダー描画
            (
                sum_ph.markdown(
                    f"**💬 代表質問：**\n\n{st.session_state.get(sum_key,'')}"
                )
                if st.session_state.get(sum_key)
                else sum_ph.empty()
            )

            (
                ans_ph.markdown(
                    f"**📝 模範回答：**\n\n{st.session_state.get(ans_key,'')}"
                )
                if st.session_state.get(ans_key)
                else ans_ph.empty()
            )

            # Markdown DL
            md = ""
            if st.session_state.get(sum_key):
                md += f"### クラスタ {cid} 代表質問\n\n{st.session_state[sum_key]}\n\n"
            if st.session_state.get(ans_key):
                md += f"### クラスタ {cid} 模範回答\n\n{st.session_state[ans_key]}\n"
            st.download_button(
                "📄 Markdown ダウンロード",
                data=md or "生成物がありません。",
                file_name=f"cluster{cid}_qa.md",
                mime="text/markdown",
                key=f"dl_{cid}",
            )
