# 🎯 Sentiment Prediction Studio

A Streamlit web application for single-review and batch CSV sentiment prediction powered by `sentiment_pipeline.joblib`, styled with custom CSS and a dark slate theme.

## Label Mapping

| Class | Sentiment |
| :---: | :--- |
| **1** | Positive |
| **0** | Negative |

## Project Structure

```text
├── app.py                     # Streamlit frontend & prediction logic
├── sentiment_pipeline.joblib  # Pre-trained scikit-learn / NLP pipeline
├── requirements.txt           # Dependencies for local & Streamlit Cloud
└── README.md                  # Project documentation
