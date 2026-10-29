import streamlit as st
import emoji
from transformers import pipeline

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Sentiment Analyzer")
st.write(
    "Analyze the emotion behind your text and emojis "
    "using an AI-powered sentiment model."
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@st.cache_resource
def load_model():
    """Load the public Hugging Face model once per app session."""
    return pipeline(
        task="text-classification",
        model=MODEL_NAME,
        device=-1,  # CPU, suitable for basic free hosting
    )


def process_emojis(text: str) -> str:
    """Convert emojis into descriptive words for the model."""
    converted_text = emoji.demojize(text, delimiters=(" ", " "))
    converted_text = converted_text.replace("_", " ")
    converted_text = converted_text.replace(":", " ")
    return " ".join(converted_text.split())


def find_emojis(text: str) -> list[str]:
    """Return emojis found in the original message."""
    return [item["emoji"] for item in emoji.emoji_list(text)]


def analyze_sentiment(text: str) -> dict:
    model = load_model()
    processed_text = process_emojis(text)

    result = model(
        processed_text,
        truncation=True,
        max_length=512,
    )[0]

    label = result["label"].upper()
    confidence = float(result["score"])

    if "POSITIVE" in label:
        sentiment = "Positive"
        message = "The message expresses a positive sentiment."
    elif "NEGATIVE" in label:
        sentiment = "Negative"
        message = "The message expresses a negative sentiment."
    else:
        sentiment = "Neutral"
        message = "The message expresses a relatively neutral sentiment."

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "message": message,
        "emojis": find_emojis(text),
        "processed_text": processed_text,
    }


st.subheader("Enter your message")

sample = st.selectbox(
    "Try an example",
    [
        "Write your own message",
        "Positive: Today was amazing 😍🔥",
        "Negative: I am so disappointed 😡",
        "Neutral: I went to the office today 🙂",
        "Mixed: Great service but terrible delivery 😕",
    ],
)

sample_messages = {
    "Positive: Today was amazing 😍🔥": "Today was amazing 😍🔥",
    "Negative: I am so disappointed 😡": "I am so disappointed 😡",
    "Neutral: I went to the office today 🙂": "I went to the office today 🙂",
    "Mixed: Great service but terrible delivery 😕":
        "Great service but terrible delivery 😕",
}

user_text = st.text_area(
    "Text and emojis",
    value=sample_messages.get(sample, ""),
    height=150,
    placeholder="Example: This product is excellent 😍",
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text or emojis first.")
    else:
        try:
            with st.spinner("Loading the model and analyzing your message..."):
                result = analyze_sentiment(user_text)

            st.divider()
            st.subheader("Analysis Results")
            st.metric("Detected sentiment", result["sentiment"])
            st.progress(result["confidence"])
            st.write(f"Model confidence: {result['confidence']:.1%}")
            st.info(result["message"])

            if result["emojis"]:
                st.write("**Emojis detected:**")
                st.write(" ".join(result["emojis"]))
            else:
                st.write("No emojis detected.")

            with st.expander("View emoji-aware text"):
                st.write(result["processed_text"])

        except Exception as error:
            st.error(
                "The analysis failed. Check your internet connection "
                "and model availability, then try again."
            )
            st.caption(f"Technical details: {error}")

st.divider()
st.caption(
    "Powered by Hugging Face Transformers. Predictions are estimates "
    "and may not correctly interpret sarcasm, slang, or mixed emotions."
)
