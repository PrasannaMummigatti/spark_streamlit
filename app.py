import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# Page
# --------------------------------------------------

st.set_page_config(
    page_title="Spark-X2.5-4B",
    page_icon="🤖"
)

st.title("🤖 Spark-X2.5-4B")

st.caption("Running locally inside Streamlit Cloud")


# --------------------------------------------------
# Model
# --------------------------------------------------

@st.cache_resource(show_spinner="Downloading and loading Spark-X2.5-4B...")
def load_model():

    model_path = hf_hub_download(
        repo_id="abenzerps/Spark-X2.5-4B-GGUF",
        filename="Spark-X2.5-4B-Q4_K_M.gguf"
    )

    llm = Llama(
        model_path=model_path,

        # Keep this modest for Streamlit Cloud
        n_ctx=4096,

        # CPU inference
        n_gpu_layers=0,

        # CPU threads
        n_threads=4,

        verbose=False
    )

    return llm


# --------------------------------------------------
# Load model
# --------------------------------------------------

try:

    llm = load_model()

except Exception as e:

    st.error("Model failed to load.")

    st.exception(e)

    st.stop()


# --------------------------------------------------
# Chat
# --------------------------------------------------

query = st.text_area(
    "Enter your question",
    placeholder="Ask Spark-X2.5-4B something..."
)


if st.button("Generate", type="primary"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Spark is thinking..."):

            try:

                response = llm.create_chat_completion(

                    messages=[
                        {
                            "role": "user",
                            "content": query
                        }
                    ],

                    temperature=0.7,

                    max_tokens=512
                )

                answer = response["choices"][0]["message"]["content"]

                st.markdown("### Response")

                st.write(answer)

            except Exception as e:

                st.error("Generation failed.")

                st.exception(e)