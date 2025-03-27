from google.cloud import bigquery

class BigQueryClient:
    def __init__(self):
        self.client = bigquery.Client()
    
    def query_data(self, query: str) -> list:
        """
        Ejecuta una consulta en BigQuery y retorna los resultados
        """
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            return [dict(row) for row in results]
        except Exception as e:
            return f"Error ejecutando la consulta: {str(e)}" 