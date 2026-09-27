import streamlit as st
import torch
from transformers import pipeline

MODEL_ID = "XHToken/Spark-X2.5-4B"

st.set_page_config(
    page_title="Spark AI",
    page_icon="🤖"
)

st.title("🤖 Spark AI")

st.write("Python / Torch environment loaded")

st.write(f"PyTorch: {torch.__version__}")
st.write(f"CUDA available: {torch.cuda.is_available()}")


@st.cache_resource
def load_model():

    st.write("⏳ Starting Spark model loading...")

    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        trust_remote_code=True,
        device=-1
    )

    st.write("✅ Spark model loaded")

    return pipe


pipe = load_model()

st.success("Spark AI is ready!")

query = st.text_area(
    "Ask Spark",
    placeholder="Enter your question...",
    height=100
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
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )

        generated = result[0]["generated_text"]

        if isinstance(generated, list):
            answer = generated[-1]["content"]
        else:
            answer = generated

        st.markdown("### Response")
        st.write(answer)