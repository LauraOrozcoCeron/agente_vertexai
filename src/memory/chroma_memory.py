from chromadb import Client, Settings
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Any
import json

class ChromaMemory:
    def __init__(self, collection_name: str = "taxi_chat_history"):
        # Crear directorio para la base de datos si no existe
        os.makedirs("./data/chroma_db", exist_ok=True)
        
        # Inicializar el cliente de Chroma con configuración local
        self.client = Client(Settings(
            chroma_db_impl="duckdb+parquet",  # Usar implementación local
            persist_directory="./data/chroma_db",
            anonymized_telemetry=False
        ))
        
        # Usar sentence-transformers para embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Crear o obtener la colección
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Usar distancia coseno para similitud
            )
        except Exception as e:
            print(f"Error inicializando colección: {str(e)}")
            # Intentar crear una nueva colección si hay error
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
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
            
            # Persistir cambios
            self.client.persist()
        except Exception as e:
            print(f"Error agregando a memoria: {str(e)}")
    
    def get_relevant_history(self, query: str, n_results: int = 3) -> List[Dict[str, str]]:
        """Obtiene el historial relevante basado en la consulta actual"""
        try:
            if not query:
                return []
                
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )
            
            # Formatear los resultados para el modelo
            relevant_history = []
            if results and results["documents"] and results["documents"][0]:
                for doc in results["documents"][0]:  # [0] porque query_texts es una lista
                    # Separar pregunta y respuesta
                    q_and_a = doc.split("\nA: ")
                    if len(q_and_a) == 2:
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
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            self.client.persist()
        except Exception as e:
            print(f"Error limpiando memoria: {str(e)}") 