from google.cloud import bigquery
from typing import List, Dict, Any
import os

class BigQueryClient:
    def __init__(self):
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS no está configurada")
            
        self.client = bigquery.Client()
        self.project_id = os.getenv('BQ_PROJECT_ID')
        self.dataset_id = os.getenv('BQ_DATASET_ID')
        self.table_id = os.getenv('BQ_TABLE_ID')
        
        if not all([self.project_id, self.dataset_id, self.table_id]):
            raise ValueError("Variables de ambiente BQ_PROJECT_ID, BQ_DATASET_ID y BQ_TABLE_ID son requeridas")
    
    def get_table_schema(self) -> List[str]:
        """Obtiene el esquema de la tabla"""
        try:
            table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            table = self.client.get_table(table_ref)
            return [field.name for field in table.schema]
        except Exception as e:
            print(f"Error obteniendo el esquema: {str(e)}")
            return []
    
    def query_data(self, query: str) -> List[Dict[str, Any]]:
        """Ejecuta una consulta en BigQuery y retorna los resultados"""
        try:
            # Limpia la consulta SQL
            query = query.strip()
            if query.startswith('sql'):
                query = query[3:].strip()
            
            # Valida que la consulta comience con SELECT
            if not query.upper().startswith('SELECT'):
                return "Error: La consulta debe comenzar con SELECT"
            
            # Asegura que la tabla esté completamente calificada
            table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            if self.table_id in query and f"{self.dataset_id}.{self.table_id}" not in query:
                query = query.replace(self.table_id, table_ref)
            
            query_job = self.client.query(query)
            results = query_job.result()
            return [dict(row) for row in results]
        except Exception as e:
            return f"Error ejecutando la consulta: {str(e)}" 