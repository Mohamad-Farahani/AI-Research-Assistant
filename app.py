import streamlit as st
import requests

st.title("📚 AI Research Assistant")

user_input = st.text_input("Ask a question:")

if user_input:
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": user_input}],
            "max_tokens": 200
        }
    )

    answer = response.json()["choices"][0]["message"]["content"]
    st.write(answer)
