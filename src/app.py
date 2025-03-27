import streamlit as st
from agent.chat_agent import GeminiAgent
from data.bigquery_client import BigQueryClient

def main():
    st.title("💬 Chatbot con Gemini y BigQuery")
    
    # Inicializar el agente y el cliente de BigQuery
    bq_client = BigQueryClient()
    agent = GeminiAgent(bq_client)
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("¿Qué deseas saber?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = agent.get_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 