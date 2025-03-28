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
            #model="gemini-1.5-pro",
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
        self.system_prompt = f"""Eres un analista especializado en datos de taxis de Nueva York que responde de manera rápida y precisa.
        
        Base de datos disponible: {self.bq_client.project_id}.{self.bq_client.dataset_id}.{self.bq_client.table_id}
        
        IMPORTANTE - UNIDADES Y CAMPOS:
        1. 🚕 Distancias (trip_distance):
           - Están en MILLAS
           - Filtrar trip_distance > 0
           - Filtrar trip_distance < 100 para excluir valores atípicos
        
        2. 💰 Tarifas y Pagos:
           - fare_amount: tarifa base en USD
           - tip_amount: propina en USD
           - total_amount: pago total en USD
        
        3. ⏰ Campos de Tiempo:
           - pickup_datetime: fecha y hora de inicio
           - dropoff_datetime: fecha y hora de fin
           - EXTRACT(HOUR FROM pickup_datetime) para hora del día
           - EXTRACT(DAYOFWEEK FROM pickup_datetime) para día de la semana (1=Domingo)
        
        EJEMPLOS DE CONSULTAS CORRECTAS:

        1. Para tarifas por hora:
        ```sql
        SELECT 
            EXTRACT(HOUR FROM pickup_datetime) as hora,
            COUNT(*) as total_viajes,
            ROUND(AVG(fare_amount), 2) as tarifa_promedio,
            ROUND(AVG(tip_amount), 2) as propina_promedio
        FROM {self.bq_client.project_id}.{self.bq_client.dataset_id}.{self.bq_client.table_id}
        WHERE pickup_datetime >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        GROUP BY hora
        ORDER BY hora
        ```

        2. Para análisis por día:
        ```sql
        SELECT 
            CASE EXTRACT(DAYOFWEEK FROM pickup_datetime)
                WHEN 1 THEN 'Domingo'
                WHEN 2 THEN 'Lunes'
                WHEN 3 THEN 'Martes'
                WHEN 4 THEN 'Miércoles'
                WHEN 5 THEN 'Jueves'
                WHEN 6 THEN 'Viernes'
                WHEN 7 THEN 'Sábado'
            END as dia_semana,
            COUNT(*) as total_viajes,
            ROUND(AVG(trip_distance), 2) as distancia_promedio,
            ROUND(AVG(total_amount), 2) as ingreso_promedio
        FROM {self.bq_client.project_id}.{self.bq_client.dataset_id}.{self.bq_client.table_id}
        GROUP BY dia_semana
        ORDER BY MIN(EXTRACT(DAYOFWEEK FROM pickup_datetime))
        ```

        3. Para distancias máximas:
        ```sql
        SELECT 
            ROUND(trip_distance, 2) as distancia_millas,
            ROUND(fare_amount, 2) as tarifa_usd,
            pickup_datetime,
            EXTRACT(HOUR FROM pickup_datetime) as hora
        FROM {self.bq_client.project_id}.{self.bq_client.dataset_id}.{self.bq_client.table_id}
        WHERE trip_distance > 0 AND trip_distance < 100
        ORDER BY trip_distance DESC
        LIMIT 5
        ```

        REGLAS IMPORTANTES:
        1. SIEMPRE usa el nombre completo de la tabla
        2. NO uses LIMIT en medio de la consulta, solo al final
        3. Usa GROUP BY con los campos exactos del SELECT
        4. ROUND los valores numéricos a 2 decimales
        5. Incluye filtros WHERE apropiados
        
        NO agregues ningún texto adicional antes o después de la consulta SQL.
        """

        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 3  # Reducido de 5 a 3 para el modelo flash-lite
    
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
            
            # Extraer la consulta SQL entre backticks o sql
            if "```" in query:
                parts = query.split("```")
                for part in parts:
                    if "SELECT" in part.upper():
                        query = part.strip()
                        break
            
            # Remover prefijo 'sql' si existe
            if query.lower().startswith('sql'):
                query = query[3:].strip()
            
            # Validar que la consulta comience con SELECT
            if not query.upper().strip().startswith('SELECT'):
                return "Error: La consulta debe comenzar con SELECT"
            
            # Asegurarse de que la consulta tenga un LIMIT
            if "LIMIT" not in query.upper():
                query += " LIMIT 1000"
            
            # Ejecutar la consulta
            results = self.bq_client.query_data(query)
            if isinstance(results, str):  # Es un mensaje de error
                return f"Error en la consulta: {results}"
            
            # Formatea los resultados para mejor legibilidad
            if not results:
                return "La consulta no retornó resultados"
            
            # Formatea los resultados de manera más amigable
            formatted_results = []
            for row in results[:5]:
                formatted_row = {}
                for key, value in row.items():
                    # Formatea números decimales a 2 lugares
                    if isinstance(value, float):
                        if 'distance' in key.lower():
                            formatted_row[key] = f"{value:.2f} millas"
                        elif any(word in key.lower() for word in ['fare', 'amount', 'total', 'tip']):
                            formatted_row[key] = f"${value:.2f}"
                        else:
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
            content = response.content
            
            # Extraer y ejecutar la consulta SQL
            sql_query = content
            results = self._execute_query(sql_query)
            
            if isinstance(results, str) and "Error" in results:
                return f"📊 {results}\n📝 Por favor, reformula tu pregunta para obtener la información deseada."
            
            # Generar interpretación de resultados
            interpretation_prompt = f"""
            Como analista de datos de taxis de NY, interpreta estos resultados: {results}
            
            REGLAS:
            1. DEBES usar EXACTAMENTE este formato:
            📊 [Dato principal con números y unidades]
            📝 [Contexto o explicación relevante]
            
            2. Usa las unidades correctas:
               - Distancias en MILLAS
               - Dinero en USD
               - Tiempo en horas/minutos
               - Conteos en número de viajes
            
            3. La explicación debe ser informativa pero breve
            
            Ejemplos de formato:
            📊 El promedio de viajes los lunes es 15,230 viajes
            📝 Esto representa un 20% más que los fines de semana
            
            📊 La tarifa promedio en hora pico es $25.50 USD
            📝 Las tarifas son 30% más altas entre 5-7 PM
            
            📊 La distancia promedio es 3.2 millas por viaje
            📝 90% de los viajes son menores a 5 millas
            """
            
            interpretation = self.llm.invoke([
                {"role": "system", "content": "Eres un analista experto en datos de taxis de NY. Tus interpretaciones son siempre precisas y útiles."},
                {"role": "user", "content": interpretation_prompt}
            ])
            
            final_response = interpretation.content
            if not final_response.startswith("📊"):
                final_response = f"📊 {final_response}"
            
            # Agregar al historial y memoria
            self.conversation_history.append({"role": "assistant", "content": final_response})
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
            self.persistent_memory.add_interaction(
                question=query,
                answer=final_response,
                metadata={"timestamp": str(datetime.now())}
            )
            
            return final_response
            
        except Exception as e:
            if "429" in str(e):
                sleep(1)
                raise
            return f"📊 Error: {str(e)}\n📝 Por favor, intenta de nuevo con una pregunta diferente."

    def clear_history(self):
        """Limpiar todo el historial"""
        self.conversation_history = []
        self.persistent_memory.clear_memory() 