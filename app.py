import streamlit as st
import torch
from transformers import pipeline

MODEL_ID = "XHToken/Spark-X2.5-4B"

st.set_page_config(
    page_title="Spark AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Spark AI")

@st.cache_resource
def load_model():

    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        trust_remote_code=True,
        device=device
    )

    return pipe


# Load model once
pipe = load_model()

# User input
query = st.text_area(
    "Ask something",
    placeholder="Enter your question...",
    height=120
)

if st.button("Generate", type="primary"):

    if not query.strip():
        st.warning("Please enter a question.")
    else:

        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        with st.spinner("Thinking..."):

            result = pipe(
                messages,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )

        # Extract assistant response
        generated = result[0]["generated_text"]

        if isinstance(generated, list):
            answer = generated[-1]["content"]
        else:
            answer = generated

        st.markdown("### Response")
        st.write(answer)