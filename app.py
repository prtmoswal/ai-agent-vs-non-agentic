import streamlit as st
from transformers import pipeline
import os

# --- Streamlit UI Configuration (MUST be the first Streamlit command) ---
st.set_page_config(page_title="AI Agent Comparison", layout="wide")

# --- Configuration for Offline Model (IMPORTANT for offline functionality) ---
# Ensure the model is downloaded locally beforehand, or accessible via a local path.
# We'll use 'distilgpt2' for its small size and CPU compatibility.
OFFLINE_MODEL_NAME = "distilgpt2"

# --- Initialize the LLM (designed for offline/CPU execution) ---
@st.cache_resource # Cache the model loading for performance
def load_offline_llm():
    """
    Loads the pre-trained language model.
    This function leverages the Hugging Face `pipeline` for text generation.
    It's configured to run on CPU ('cpu') to ensure offline capability without GPU.
    """
    # Moved st.write and st.success calls outside of this cached function
    # or ensure they are only called conditionally after st.set_page_config
    try:
        # For true offline, ensure 'OFFLINE_MODEL_NAME' points to a pre-downloaded model directory
        generator = pipeline(
            'text-generation',
            model=OFFLINE_MODEL_NAME,
            device='cpu' # Explicitly use CPU for offline execution
        )
        return generator
    except Exception as e:
        st.error(f"Error loading LLM model. Ensure '{OFFLINE_MODEL_NAME}' is available locally or check internet connection if not pre-downloaded: {e}")
        st.stop() # Stop the app if the model cannot be loaded

# Call load_offline_llm after set_page_config
llm_generator = load_offline_llm()

# Display loading status after set_page_config and model loading
st.info(f"Model {OFFLINE_MODEL_NAME} loaded successfully for offline use!")


# --- Define a simple 'Tool' for our Agent ---
def simple_calculator(expression: str) -> str:
    """
    A simple calculator tool. Evaluates a mathematical expression.
    This simulates an agent's ability to call external functions.
    """
    try:
        result = eval(expression) # WARNING: eval() can be dangerous with untrusted input for production
        return f"The result of '{expression}' is: {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"

# --- Agent Logic (Simplified) ---
def run_agent(query: str):
    """
    Simulates a basic agent that tries to understand the query and
    potentially use a tool.
    """
    processing_info = [] # To capture agent's internal messages

    if "calculate" in query.lower() or ("what is" in query.lower() and any(op in query for op in ["+", "-", "*", "/"])):
        processing_info.append("Agent detected a calculation request. Attempting to use calculator tool...")
        expression = query.lower().replace("calculate", "").replace("what is", "").strip()
        expression = expression.replace("?", "").replace("the result of", "").strip()
        response = simple_calculator(expression)
    else:
        processing_info.append("Agent generating response using LLM...")
        response_data = llm_generator(
            query,
            max_new_tokens=50,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        response = response_data[0]['generated_text']

    return "\n".join(processing_info) + "\n\n" + response

# --- Non-Agentic LLM Logic ---
def run_non_agentic_llm(query: str):
    """
    Directly runs the LLM for text generation without any agentic logic or tools.
    """
    processing_info = [] # To capture non-agentic's internal messages
    processing_info.append("Directly generating response using LLM (no tools)..")
    response_data = llm_generator(
        query,
        max_new_tokens=100, # Longer response for non-agentic
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    return "\n".join(processing_info) + "\n\n" + response_data[0]['generated_text']

# --- Main App Title and Description ---
st.title("AI Agentic vs. Non-Agentic: A Side-by-Side Demo")
st.markdown("Enter a prompt below and see how a simple **Non-Agentic AI** (direct LLM output) compares to an **Agentic AI** (LLM with a calculator tool).")

# Example Prompts
example_prompts = [
    "Select a prompt from here...",
    "What is 15 * 3?", # Math query
    "Tell me a short story about a brave knight.", # Creative query
    "Calculate (20 + 5) / 5.", # More complex math query
    "Explain the concept of artificial intelligence.", # Explanatory query
    "What is 1000 - 123?", # Math query
]

selected_prompt = st.selectbox("Choose an example prompt or type your own:", example_prompts, index=0)
user_input = st.text_input("Or type your own query here:", value="" if selected_prompt == "Select a prompt from here..." else selected_prompt, key="user_input_combined")

if st.button("Run Both AIs"):
    if user_input:
        col1, col2 = st.columns(2)

        with st.spinner("Non-Agentic AI thinking..."):
            non_agentic_response = run_non_agentic_llm(user_input)

        with st.spinner("Agentic AI thinking..."):
            agentic_response = run_agent(user_input)

        with col1:
            st.subheader("Non-Agentic AI Response")
            st.info(non_agentic_response) # Use st.info for a distinct block

        with col2:
            st.subheader("Agentic AI Response")
            st.success(agentic_response) # Use st.success for a distinct block
    else:
        st.warning("Please enter a query or select an example prompt.")

st.markdown("---")

# --- Comparison Section (Expandable) ---
with st.expander("Click to see the detailed comparison: Non-Agentic vs. Agentic AI"):
    st.markdown("""
    ## Non-Agentic AI vs. Agentic AI: A Comparison

    The terms "non-agentic AI" and "agentic AI" describe different paradigms for how an AI system processes information and interacts with its environment. The core distinction lies in the AI's **ability to reason, plan, and autonomously execute actions (often using external tools) to achieve a goal**, rather than simply generating a response based on its direct training data.

    | Feature             | Non-Agentic AI (e.g., Basic LLM App)                   | Agentic AI (e.g., LLM-Powered Agent)                     |
    | :------------------ | :------------------------------------------------------- | :--------------------------------------------------------- |
    | **Core Function** | Primarily focuses on **generating content** (text, code, images) or **answering questions** based on its internal knowledge. | Aims to **achieve a goal** by reasoning, planning, and executing actions, often involving external tools. |
    | **Decision-Making** | Directly processes input and generates output. No inherent "decision-making" on *how* to answer or *what action to take*. | Possesses a **decision-making layer** that determines the best approach to fulfill a request. It can decide *if* and *when* to use a tool. |
    | **Tool Use** | **No ability** to use external tools (e.g., calculators, databases, APIs, web browsers). Limited to its own "knowledge." | **Can leverage external tools** to extend its capabilities beyond pure generation, providing accurate, real-time, or specific information/actions. |
    | **Workflow** | **Input -> LLM -> Output** | **Input -> Agent (with LLM) -> Reasoning/Planning -> Tool Use (Optional) -> Output** |
    | **Versatility** | Limited to tasks that can be solved purely through text generation/completion. | Highly versatile; can perform complex, multi-step tasks by combining reasoning with tool capabilities. |
    | **Accuracy/Factuality** | Reliant on its training data; prone to "hallucinations" or providing outdated information. | Can enhance factual accuracy by using tools to retrieve real-time or specific data, reducing reliance on memorized information. |
    | **Complexity** | Simpler to design and implement for single-task purposes. | More complex to design due to the need for planning, tool integration, and robust error handling. |
    | **Example Queries** | "Write a poem about the ocean."<br>"Explain quantum physics." | "What is the current stock price of Google?" (uses web search tool)<br>"Calculate 25% of 340." (uses calculator tool)<br>"Summarize this document and then email it to John." (uses document processing and email tools) |
    """)

st.markdown("""
---
**How this App Works Offline/Agentic:**
* **Offline:** The `distilgpt2` model is loaded directly to your CPU via the `transformers` library, meaning no internet connection is required after the initial model download. The `@st.cache_resource` decorator ensures it loads only once.
* **Agentic (Simplified):** The 'Agentic AI' path includes a basic 'calculator' tool. It uses simple logic to detect if your query looks like a calculation. If so, it calls the `simple_calculator` function (its 'tool'); otherwise, it uses the LLM for text generation. A more advanced agent would use the LLM itself to decide which tools to call.
* **Non-Agentic:** The 'Non-Agentic AI' path directly passes all user input to the LLM for text generation without any intervening logic or tool use.
* **Streamlit:** Provides the interactive web interface.
""")
