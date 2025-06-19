import streamlit as st
import pandas as pd
import openai
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np

st.set_page_config(
    page_title="講義の質疑応答まとめアプリ : AIE Proj 04", layout="centered"
)
st.title("📚 講義の質疑応答まとめアプリ : AIE Proj 04")

# Streamlit Cloud Secrets から OpenAI APIキーを取得
openai_api_key = st.secrets["openai_api_key"]

# OpenAI クライアントのセットアップ（openai>=1.0.0対応）
client = openai.OpenAI(api_key=openai_api_key)

# クラスタ数の指定
num_clusters = st.slider("クラスタ数（KMeans）", min_value=2, max_value=20, value=5)

# ファイルアップロード
uploaded_file = st.file_uploader(
    "📤 CSVファイルをアップロード（質問, 回答の列を含む）", type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "質問" not in df.columns:
        st.error("'質問' という列が必要です。CSVを確認してください。")
    else:
        st.success(f"✅ {len(df)} 件の質問を読み込みました。")
        questions = df["質問"].dropna().tolist()

        # ベクトル化（OpenAI埋め込み）
        with st.spinner("💠 OpenAI埋め込みを取得中..."):
            try:
                embeddings_response = client.embeddings.create(
                    model="text-embedding-ada-002", input=questions
                )
                vectors = np.array([e.embedding for e in embeddings_response.data])
            except Exception as e:
                st.error(f"埋め込み取得に失敗しました: {e}")
                st.stop()

        # KMeansクラスタリング
        vectors_norm = normalize(vectors)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(vectors_norm)
        df["クラスタ"] = labels

        st.success(
            f"✅ {num_clusters} クラスタに分類しました。各クラスタごとに要約と回答を生成できます。"
        )

        # クラスタごとに要約と回答生成
        for cluster_id in range(num_clusters):
            cluster_df = df[df["クラスタ"] == cluster_id]
            cluster_questions = cluster_df["質問"].tolist()

            with st.expander(
                f"▶️ クラスタ {cluster_id}：{len(cluster_questions)} 件の質問"
            ):
                st.markdown("\n".join([f"- {q}" for q in cluster_questions]))

                if st.button(f"🧠 クラスタ {cluster_id} の代表質問を生成"):
                    prompt = """
以下の質問は講義中に受けた似た内容の質問です。これらを要約して、代表的な1つの質問にまとめてください。
質問一覧：
"""
                    prompt += "\n".join([f"- {q}" for q in cluster_questions])
                    prompt += "\n\n代表質問："

                    with st.spinner("GPTが代表質問を要約中..."):
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
                            summary_question = response.choices[
                                0
                            ].message.content.strip()
                            st.markdown(f"**💬 代表質問：** {summary_question}")

                            if st.button(f"💡 クラスタ {cluster_id} の模範回答を生成"):
                                answer_prompt = f"以下の質問に対して、講義で使える模範的な回答を生成してください。\n\n質問：{summary_question}\n\n回答："
                                with st.spinner("GPTが模範回答を生成中..."):
                                    try:
                                        answer_response = client.chat.completions.create(
                                            model="gpt-4",
                                            messages=[
                                                {
                                                    "role": "system",
                                                    "content": "あなたは親切でわかりやすく説明できる講義アシスタントです。",
                                                },
                                                {
                                                    "role": "user",
                                                    "content": answer_prompt,
                                                },
                                            ],
                                            temperature=0.7,
                                        )
                                        model_answer = answer_response.choices[
                                            0
                                        ].message.content.strip()
                                        st.markdown(
                                            f"**📝 模範回答：**\n\n{model_answer}"
                                        )
                                    except Exception as e:
                                        st.error(
                                            f"模範回答生成中にエラーが発生しました: {e}"
                                        )
                        except Exception as e:
                            st.error(f"代表質問生成中にエラーが発生しました: {e}")
