import streamlit as st
from transformers import pipeline

MODEL_ID = "XHToken/Spark-X2.5-4B"


st.set_page_config(
    page_title="Spark AI",
    page_icon="🤖"
)

st.title("🤖 Spark AI")


@st.cache_resource
def load_model():

    model = pipeline(
        "text-generation",
        model=MODEL_ID,
        trust_remote_code=True,
        device=-1
    )

    return model


try:

    pipe = load_model()

    query = st.text_area(
        "Ask Spark",
        placeholder="Enter your question...",
        height=100
    )

    if st.button("Generate", type="primary"):

        if query.strip():

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

        else:

            st.warning("Please enter a question.")


except Exception as e:

    st.error("Model failed to load.")
    st.exception(e)