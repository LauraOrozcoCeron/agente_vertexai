from dotenv import load_dotenv
import os
import streamlit as st
from agent.chat_agent import GeminiAgent

# Cargar variables de entorno
load_dotenv()

def main():
    st.title("🚕 Asistente de Análisis de Taxis NY")
    
    # Sidebar con información
    st.sidebar.title("Información")
    st.sidebar.info("""
    Este asistente puede responder preguntas sobre los datos de taxis amarillos de Nueva York.
    
    Ejemplos de preguntas:
    - ¿Cuál es el promedio de tarifa por viaje?
    - ¿Cuántos viajes se realizaron ayer?
    - ¿Cuáles son las zonas más populares?
    """)
    
    # Botones en el sidebar
    st.sidebar.title("Controles")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Limpiar Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Limpiar Memoria"):
            if "agent" in st.session_state:
                st.session_state.agent.clear_history()
            st.session_state.messages = []
            st.rerun()
    
    # Mostrar información sobre la memoria
    if "agent" in st.session_state:
        memory_size = len(st.session_state.agent.persistent_memory.collection.get()["ids"])
        st.sidebar.info(f"Interacciones en memoria: {memory_size}")
    
    # Inicializar el agente
    if "agent" not in st.session_state:
        st.session_state.agent = GeminiAgent()
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("¿Qué deseas saber sobre los taxis de NY?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analizando datos..."):
                response = st.session_state.agent.get_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 