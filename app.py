import streamlit as st
import torch
from transformers import pipeline

MODEL_ID = "XHToken/Spark-X2.5-4B"

st.set_page_config(
    page_title="Spark AI",
    page_icon="🤖"
)

st.title("🤖 Spark AI")


@st.cache_resource
def load_model():

    st.write("Loading Spark-X2.5-4B...")

    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        trust_remote_code=True,
        device="cpu"
    )

    return pipe


try:

    pipe = load_model()

    query = st.text_area(
        "Enter your question",
        placeholder="Ask Spark anything...",
        height=120
    )

    if st.button("Generate"):

        if not query.strip():
            st.warning("Please enter a question.")

        else:

            messages = [
                {
                    "role": "user",
                    "content": query
                }
            ]

            with st.spinner("Generating response..."):

                output = pipe(
                    messages,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )

            generated = output[0]["generated_text"]

            if isinstance(generated, list):

                answer = generated[-1]["content"]

            else:

                answer = generated

            st.markdown("### Response")
            st.write(answer)


except Exception as e:

    st.error("Model failed to load.")

    st.exception(e)