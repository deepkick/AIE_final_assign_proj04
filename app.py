import streamlit as st
import pandas as pd
import openai
import os

st.set_page_config(page_title="Q&Aまとめアプリ", layout="centered")
st.title("📚 質疑応答の要約アプリ")

# Streamlit Cloud Secrets から OpenAI APIキーを取得
openai_api_key = st.secrets["openai_api_key"]

# OpenAI クライアントのセットアップ（openai>=1.0.0対応）
client = openai.OpenAI(api_key=openai_api_key)

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

        if st.button("▶️ GPTで代表質問を要約する"):
            prompt = """
以下の質問は講義中に受けた似た内容の質問です。これらを要約して、代表的な1つの質問にまとめてください。
質問一覧：
"""
            questions = df["質問"].dropna().tolist()
            prompt += "\n".join(
                [f"- {q}" for q in questions[:20]]
            )  # 最初の20件でテスト
            prompt += "\n\n代表質問："

            with st.spinner("GPTが要約中..."):
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
                    st.subheader("💬 代表質問（要約）")
                    st.markdown(summary_question)

                    # 🔢 トークン使用量の表示
                    usage = response.usage
                    if usage:
                        total_tokens = usage.total_tokens
                        st.info(f"🔢 トークン消費量: {total_tokens} tokens")

                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
