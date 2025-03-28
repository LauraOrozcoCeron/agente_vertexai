from langchain_google_genai import ChatGoogleGenerativeAI
import os
from time import sleep
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
from typing import List, Dict
from data.bigquery_client import BigQueryClient
from memory.chroma_memory import ChromaMemory
from datetime import datetime

class GeminiAgent:
    def __init__(self):
        # Asegurarse de que la API key esté configurada
        if 'GOOGLE_API_KEY' not in os.environ:
            raise ValueError("GOOGLE_API_KEY no está configurada en las variables de entorno")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",  # Cambiado a flash-lite
            temperature=0.5,  # Reducida para respuestas más precisas
            convert_system_message_to_human=True,
            max_output_tokens=150,  # Reducido para respuestas más concisas
            top_p=0.8,  # Añadido para mejor control de la generación
            top_k=40  # Añadido para mejor control de la generación
        )
        
        # Inicializar cliente de BigQuery
        self.bq_client = BigQueryClient()
        
        # Obtener el esquema de la tabla
        self.table_schema = self.bq_client.get_table_schema()
        
        # Inicializar memoria persistente
        self.persistent_memory = ChromaMemory()
        
        # Modificar el system prompt para incluir contexto del historial
        self.system_prompt = f"""Eres un analista de datos conciso que analiza datos de taxis de NY.  
        Tabla disponible: {self.bq_client.project_id}.{self.bq_client.dataset_id}.{self.bq_client.table_id}  
        Campos: {', '.join(self.table_schema)}  

        ### Instrucciones:
        1. Genera consultas SQL simples y eficientes  
        2. Usa SIEMPRE el nombre completo de la tabla: {self.bq_client.project_id}.{self.bq_client.dataset_id}.{self.bq_client.table_id}  
        3. Da respuestas en lenguaje natural con interpretación de los datos  
        4. Siempre responde con una oración clara que contenga el valor numérico y las unidades (USD, millas, minutos, etc.)  
        5. Contextualiza el resultado, explicando qué significa el número obtenido en términos prácticos  
        6. Si no puedes responder algo, dilo directamente
        7. Usa el historial relevante proporcionado para dar respuestas más contextualizadas

        ### Formato de respuesta:
        - Para consultas: Genera SQL entre ```  
        - Para interpretaciones: **"El [dato solicitado] es [valor] [unidades]. [Contexto adicional]."**  

        #### **Ejemplo de respuesta:**
        **"El promedio de tarifa por viaje es $14.65 USD. Este valor incluye solo la tarifa base sin propinas ni cargos adicionales."**  

        """

        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 5  # Reducido para mantener el contexto más relevante
    
    def _get_relevant_history(self) -> List[Dict[str, str]]:
        """Obtiene el historial relevante combinando memoria a corto y largo plazo"""
        # Obtener historial reciente de la memoria en RAM
        recent_history = self.conversation_history[-self.max_history:]
        
        # Obtener historial relevante de Chroma
        persistent_history = self.persistent_memory.get_relevant_history(
            query=recent_history[-1]["content"] if recent_history else "",
            n_results=3
        )
        
        # Combinar historiales y agregar el system prompt
        return ([{"role": "system", "content": self.system_prompt}] + 
                persistent_history + recent_history)
    
    def _execute_query(self, query: str) -> str:
        """Ejecuta una consulta SQL y formatea los resultados"""
        try:
            # Limpia la consulta SQL
            query = query.strip()
            if "```" in query:
                # Extrae la consulta entre los backticks
                query = query.split("```")[1].strip()
            
            results = self.bq_client.query_data(query)
            if isinstance(results, str):  # Es un mensaje de error
                return f"Error en la consulta: {results}"
            
            # Formatea los resultados para mejor legibilidad
            if not results:
                return "La consulta no retornó resultados"
            
            # Formatea los resultados de manera más amigable
            formatted_results = []
            for row in results[:5]:  # Limitamos a 5 resultados
                formatted_row = {}
                for key, value in row.items():
                    # Formatea números decimales a 2 lugares
                    if isinstance(value, float):
                        formatted_row[key] = f"{value:.2f}"
                    else:
                        formatted_row[key] = value
                formatted_results.append(formatted_row)
            
            return formatted_results
            
        except Exception as e:
            return f"Error ejecutando la consulta: {str(e)}"
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),  # Tiempos más cortos
        retry_error_callback=lambda _: "Servicio ocupado, intenta de nuevo."
    )
    def get_response(self, query: str) -> str:
        try:
            # Agregar la pregunta al historial
            self.conversation_history.append({"role": "user", "content": query})
            
            # Obtener respuesta usando el historial
            messages = self._get_relevant_history()
            response = self.llm.invoke(messages)
            
            # Verificar si la respuesta contiene una consulta SQL
            content = response.content
            if "SELECT" in content.upper():
                # Extraer la consulta SQL
                sql_start = content.find("```") + 3
                sql_end = content.find("```", sql_start)
                if sql_start >= 3 and sql_end != -1:
                    sql_query = content[sql_start:sql_end].strip()
                    
                    # Ejecutar la consulta y obtener resultados
                    results = self._execute_query(sql_query)
                    
                    # Obtener interpretación breve de los resultados
                    interpretation_prompt = f"Datos: {results}\nInterpreta en máximo 2 oraciones."
                    interpretation = self.llm.invoke([{"role": "user", "content": interpretation_prompt}])
                    content = interpretation.content
            
            # Agregar la respuesta al historial
            self.conversation_history.append({"role": "assistant", "content": content})
            
            # Mantener el historial dentro del límite
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
            # Agregar la interacción a la memoria persistente
            self.persistent_memory.add_interaction(
                question=query,
                answer=content,
                metadata={
                    "timestamp": str(datetime.now()),
                    "has_sql": "SELECT" in content.upper()
                }
            )
            
            return content
            
        except Exception as e:
            if "429" in str(e):
                sleep(1)  # Tiempo de espera reducido
                raise
            return f"Error: {str(e)}"

    def clear_history(self):
        """Limpiar todo el historial"""
        self.conversation_history = []
        self.persistent_memory.clear_memory() 