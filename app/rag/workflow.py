import logging
from typing import Literal

from chromadb import get_settings
from langchain_openai import ChatOpenAI
from openai import max_retries
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph,START,END
from app.core.config import get_Settings
from app.rag.state import AgentState,RouteDecision,EvidenceGrade
from app.rag.vectorstore import get_retriever

logger=logging.getLogger(__name__)
settings=get_Settings()

_llm=None
_web_search=None

def llm():
    global _llm
    
    if _llm is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        
        _llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key
        )
    return _llm
    
    
def web_search():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("Tavily API key is missing")
        
        _web_search=TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search


def add_trace(state:AgentState,message:str):
    return [*state.get("trace",[]),message]


def route_question(state:AgentState):
    router=llm().with_structured_output(RouteDecision,method="json_mode")
    decision=router.invoke(f"""
    You route messages for an enterprise IT Suuport assistant
    use kb for questions about company policies,VPN,Password reset,MFA,software,troubleshooting
    use direct only for greetings,thanks for casual that needs no company knowledge.
    Question : {state['question']}
    Return valid JSON like {{"route":"kb"}}.
    """)
    return {"source_used":decision.route,"trace":add_trace(state,f"Router -> {decision.route}")}