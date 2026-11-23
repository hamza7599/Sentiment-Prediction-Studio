import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sentiment Predict",
    page_icon="",
    layout="centered"
)

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top, #1e293b 0%, #0f172a 65%, #090d16 100%);
            color: #f1f5f9;
        }
        .hero-box {
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 24px 28px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
            margin-bottom: 22px;
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 700;
            color: #f8fafc;
            margin: 0 0 6px 0;
        }
        .hero-sub {
            font-size: 0.96rem;
            color: #cbd5e1;
            margin: 0;
        }
        .hero-sub code {
            background: rgba(15, 23, 42, 0.8);
            color: #fbbf24;
            padding: 2px 6px;
            border-radius: 6px;
        }
        .result-pos {
            background: rgba(16, 185, 129, 0.14);
            border: 1px solid rgba(52, 211, 153, 0.55);
            color: #a7f3d0;
            border-radius: 14px;
            padding: 18px 22px;
            margin-top: 14px;
            box-shadow: 0 6px 18px rgba(16, 185, 129, 0.16);
        }
        .result-neg {
            background: rgba(244, 63, 94, 0.14);
            border: 1px solid rgba(251, 113, 133, 0.55);
            color: #fecdd3;
            border-radius: 14px;
            padding: 18px 22px;
            margin-top: 14px;
            box-shadow: 0 6px 18px rgba(244, 63, 94, 0.16);
        }
        .result-label {
            font-size: 1.35rem;
            font-weight: 700;
            margin: 0 0 4px 0;
        }
        .result-conf {
            font-size: 0.95rem;
            opacity: 0.92;
            margin: 0;
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.25rem;
            font-weight: 600;
            box-shadow: 0 6px 16px rgba(234, 88, 12, 0.3);
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #d97706 0%, #c2410c 100%);
            color: #ffffff;
        }
        .stTextArea textarea {
            border-radius: 12px !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            background-color: #1e293b !important;
            color: #f8fafc !important;
        }
        label, .stMarkdown p {
            color: #e2e8f0;
        }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_pipeline():
    return joblib.load("sentiment_pipeline.joblib")


def map_sentiment(pred):
    return "Positive" if int(pred) == 1 else "Negative"


st.markdown(
    """
    <div class="hero-box">
        <p class="hero-title"> Sentiment Prediction </p>
        <p class="hero-sub">Class <b>1</b> → <b>Positive</b> &nbsp;|&nbsp; Class <b>0</b> → <b>Negative</b> — powered by <code>sentiment_pipeline.joblib</code></p>
    </div>
    """,
    unsafe_allow_html=True
)

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("`sentiment_pipeline.joblib` was not found in the current directory.")
    st.stop()

tab1, tab2 = st.tabs(["✍️ Single Review", "📂 Batch CSV"])

with tab1:
    text = st.text_area(
        "Enter review or text",
        height=140,
        placeholder="Write a review here..."
    )

    if st.button("Predict Sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            pred = pipeline.predict([text])[0]
            label = map_sentiment(pred)

            conf_text = ""
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba([text])[0]
                classes = list(getattr(pipeline, "classes_", [0, 1]))
                target_idx = classes.index(pred) if pred in classes else int(pred)
                conf_text = f"Confidence: {probs[target_idx]:.1%}"

            if int(pred) == 1:
                st.markdown(
                    f"""
                    <div class="result-pos">
                        <p class="result-label">😊 Positive (Class 1)</p>
                        <p class="result-conf">{conf_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-neg">
                        <p class="result-label">😞 Negative (Class 0)</p>
                        <p class="result-conf">{conf_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba([text])[0]
                classes = list(getattr(pipeline, "classes_", [0, 1]))
                prob_rows = [
                    {
                        "Class": f"{int(c)} ({map_sentiment(c)})",
                        "Probability": f"{p:.1%}"
                    }
                    for c, p in zip(classes, probs)
                ]
                st.dataframe(pd.DataFrame(prob_rows), hide_index=True, use_container_width=True)

with tab2:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        col = st.selectbox("Select text column", df.columns)

        if st.button("Run Batch Prediction", type="primary"):
            texts = df[col].fillna("").astype(str).tolist()
            preds = pipeline.predict(texts)

            df["Class"] = [int(p) for p in preds]
            df["Sentiment"] = [map_sentiment(p) for p in preds]

            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(texts)
                df["Confidence"] = [f"{max(p):.1%}" for p in probs]

            st.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV",
                data=csv_bytes,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )