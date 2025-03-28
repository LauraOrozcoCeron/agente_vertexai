from dotenv import load_dotenv
import os
import streamlit as st
from agent.chat_agent import GeminiAgent

# Cargar variables de entorno
load_dotenv()

def main():
    st.title("🚕 Análisis de Taxis NY")
    
    # Sidebar con información
    st.sidebar.title("📊 Guía de Preguntas")
    st.sidebar.info("""
    Ejemplos de preguntas que puedes hacer:
    
    💰 Tarifas y Pagos:
    - ¿Cuál es la tarifa promedio por viaje?
    - ¿Cuánto se gana en propinas en hora pico?
    - ¿Cuál es el pago total promedio por viaje?
    
    ⏰ Tiempo y Patrones:
    - ¿Cuáles son las horas más ocupadas?
    - ¿Qué día de la semana hay más viajes?
    - ¿Cuánto duran los viajes en promedio?
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