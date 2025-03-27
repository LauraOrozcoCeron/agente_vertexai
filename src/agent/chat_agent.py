from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from .prompts import AGENT_PROMPT

class GeminiAgent:
    def __init__(self, bq_client):
        self.bq_client = bq_client
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
        )
        
        self.tools = [
            Tool(
                name="BigQuery_Search",
                func=self.bq_client.query_data,
                description="Útil para buscar información en BigQuery"
            )
        ]
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=AGENT_PROMPT
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )
    
    def get_response(self, query: str) -> str:
        try:
            response = self.agent_executor.run(query)
            return response
        except Exception as e:
            return f"Lo siento, ocurrió un error: {str(e)}" 