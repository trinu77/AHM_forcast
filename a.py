

from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# --- Load API Key ---
load_dotenv(override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

SYSTEM_PROMPT = """You are a helpful assistant.
You can access only the table 'sensor_data_IEMA6012001'.
"""

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

def create_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0, api_key=GOOGLE_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages")
    ])

    def agent(state):
        return {"messages": [HumanMessage(content="Connected to Gemini and ready.")]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
