import streamlit as st
import subprocess
import sys

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖"
)

st.title("🤖 AI Agent")
st.caption("Reason → Act → Observe")

goal = st.text_area(
    "Enter your goal",
    value="Find the population of France and Germany, then calculate the combined total."
)

if st.button("Run Agent 🚀"):

    if not goal.strip():
        st.warning("Please enter a goal.")
        st.stop()

    st.write("🤖 Agent is working...")

    try:
        result = subprocess.run(
            [sys.executable, "agent.py"],
            input=goal + "\n",
            text=True,
            capture_output=True,
            timeout=60
        )

        if result.returncode != 0:
            st.error("Agent failed.")
            st.code(result.stderr)
        else:
            output = result.stdout

            st.subheader("Agent Trace")
            st.code(output)

    except subprocess.TimeoutExpired:
        st.error(
            "Agent took longer than 60 seconds and was stopped. "
            "This prevents an infinite loop and protects your Gemini quota."
        )

    except Exception as e:
        st.error(f"Error: {e}")