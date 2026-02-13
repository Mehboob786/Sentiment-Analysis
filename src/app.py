from pathlib import Path

import joblib
import streamlit as st

from responder import generate_reply


MODEL_PATH = Path("models/sentiment_model.joblib")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file missing. Run `python src/prepare_data.py` then `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict(model, text: str) -> tuple[str, float]:
    probs = model.predict_proba([text])[0]
    labels = model.classes_
    idx = probs.argmax()
    return str(labels[idx]), float(probs[idx])


def main() -> None:
    st.set_page_config(page_title="Sentiment Support Chatbot", page_icon="💬")
    st.title("Sentiment-Aware Customer Support Chatbot")
    st.caption("Rule-based support responses + ML sentiment classification")

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input("Your message")
        submitted = st.form_submit_button("Send")

    if submitted and user_msg.strip():
        sentiment, confidence = predict(model, user_msg)
        reply = generate_reply(user_msg, sentiment, confidence)
        st.session_state.history.append(("You", user_msg))
        st.session_state.history.append(
            ("Bot", f"[sentiment={sentiment}, confidence={confidence:.2f}] {reply}")
        )

    for speaker, text in reversed(st.session_state.history):
        if speaker == "You":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")


if __name__ == "__main__":
    main()
