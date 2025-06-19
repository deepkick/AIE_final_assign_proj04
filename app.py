import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np

# ──────────────────────────────────────────────
# Streamlit 基本設定
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="講義の質疑応答まとめアプリ : AIE Proj 04",
    layout="centered",
)
st.title("📚 講義の質疑応答まとめアプリ : AIE Proj 04")

# ──────────────────────────────────────────────
# OpenAI クライアント初期化
# ──────────────────────────────────────────────
openai_api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=openai_api_key)

# ──────────────────────────────────────────────
# UI：クラスタ数・ファイルアップロード
# ──────────────────────────────────────────────
num_clusters = st.slider("クラスタ数（KMeans）", min_value=2, max_value=20, value=5)

uploaded_file = st.file_uploader(
    "📤 CSVファイルをアップロード（列: 質問, 回答）", type=["csv"]
)

# ──────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "質問" not in df.columns:
        st.error("'質問' という列が必要です。CSVを確認してください。")
        st.stop()

    st.success(f"✅ {len(df)} 件の質問を読み込みました。")

    questions = df["質問"].dropna().tolist()

    # -- OpenAI Embeddings --------------------------------------------------
    with st.spinner("💠 OpenAI 埋め込みを取得中..."):
        try:
            embeddings_response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=questions,
            )
            vectors = np.array([e.embedding for e in embeddings_response.data])
        except Exception as e:
            st.error(f"埋め込み取得に失敗しました: {e}")
            st.stop()

    # -- KMeans クラスタリング ---------------------------------------------
    vectors_norm = normalize(vectors)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors_norm)
    df["クラスタ"] = labels

    st.success(
        f"✅ {num_clusters} クラスタに分類しました。各クラスタごとに要約と回答を生成できます。"
    )

    # ----------------------------------------------------------------------
    # クラスタごとの UI
    # ----------------------------------------------------------------------
    for cluster_id in range(num_clusters):
        cluster_df = df[df["クラスタ"] == cluster_id]
        cluster_questions = cluster_df["質問"].tolist()

        with st.expander(f"▶️ クラスタ {cluster_id}：{len(cluster_questions)} 件の質問"):
            summary_key = f"summary_question_{cluster_id}"
            answer_key = f"model_answer_{cluster_id}"

            # ── 既存の代表質問・模範回答を表示 ─────────────────────────────
            if summary_key in st.session_state and st.session_state[summary_key]:
                st.markdown(
                    f"""**💬 代表質問：**

{st.session_state[summary_key]}"""
                )

            if answer_key in st.session_state and st.session_state[answer_key]:
                st.markdown(
                    f"""**📝 模範回答：**

{st.session_state[answer_key]}"""
                )

            # 質問リストを列挙
            st.markdown("\n".join([f"- {q}" for q in cluster_questions]))

            # ── GPT で代表質問生成 ──────────────────────────────────────
            if st.button(
                f"🧠 クラスタ {cluster_id} の代表質問を生成",
                key=f"summary_button_{cluster_id}",
            ):
                prompt = (
                    "以下の質問は講義中に受けた似た内容の質問です。"
                    "これらを要約して、代表的な1つの質問にまとめてください。\n"
                    "質問一覧：\n"
                )
                prompt += "\n".join([f"- {q}" for q in cluster_questions])
                prompt += "\n\n代表質問："

                with st.spinner("GPT が代表質問を要約中..."):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "あなたは親切で要約が上手なアシスタントです。",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.5,
                        )
                        summary_question = response.choices[0].message.content.strip()
                        st.session_state[summary_key] = summary_question
                        st.session_state[answer_key] = ""  # 先に回答をクリア
                    except Exception as e:
                        st.error(f"代表質問生成中にエラーが発生しました: {e}")

            # ── GPT で模範回答生成 ──────────────────────────────────────
            if st.button(
                f"💡 クラスタ {cluster_id} の模範回答を生成",
                key=f"answer_button_{cluster_id}",
            ):
                summary_question = st.session_state.get(summary_key)
                if not summary_question:
                    st.error("先に代表質問を生成してください。")
                    st.stop()

                answer_prompt = (
                    "以下の質問に対して、講義で使える模範的な回答を生成してください。\n\n"
                    f"質問：{summary_question}\n\n回答："
                )

                with st.spinner("GPT が模範回答を生成中..."):
                    try:
                        answer_response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "あなたは親切でわかりやすく説明できる講義アシスタントです。",
                                },
                                {"role": "user", "content": answer_prompt},
                            ],
                            temperature=0.7,
                        )
                        model_answer = answer_response.choices[
                            0
                        ].message.content.strip()
                        st.session_state[answer_key] = model_answer
                    except Exception as e:
                        st.error(f"模範回答生成中にエラーが発生しました: {e}")
