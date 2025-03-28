from chromadb import Client, Settings
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Any
import json

class ChromaMemory:
    def __init__(self, collection_name: str = "taxi_chat_history"):
        # Inicializar el cliente de Chroma
        self.client = Client(Settings(
            persist_directory="./data/chroma_db",
            anonymized_telemetry=False
        ))
        
        # Usar sentence-transformers para embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Crear o obtener la colección
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
    
    def add_interaction(self, question: str, answer: str, metadata: Dict[str, Any] = None) -> None:
        """Agrega una interacción a la memoria"""
        try:
            # Crear un ID único basado en el timestamp
            interaction_id = str(len(self.collection.get()["ids"]) + 1)
            
            # Agregar la interacción a Chroma
            self.collection.add(
                documents=[f"Q: {question}\nA: {answer}"],
                metadatas=[metadata or {}],
                ids=[interaction_id]
            )
        except Exception as e:
            print(f"Error agregando a memoria: {str(e)}")
    
    def get_relevant_history(self, query: str, n_results: int = 3) -> List[Dict[str, str]]:
        """Obtiene el historial relevante basado en la consulta actual"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Formatear los resultados para el modelo
            relevant_history = []
            for doc in results["documents"][0]:  # [0] porque query_texts es una lista
                # Separar pregunta y respuesta
                q_and_a = doc.split("\nA: ")
                question = q_and_a[0].replace("Q: ", "")
                answer = q_and_a[1]
                
                relevant_history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ])
            
            return relevant_history
        except Exception as e:
            print(f"Error obteniendo historial: {str(e)}")
            return []
    
    def clear_memory(self) -> None:
        """Limpia toda la memoria"""
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"Error limpiando memoria: {str(e)}") 