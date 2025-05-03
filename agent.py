"""LangGraph Agent using Mistral"""
import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from transformers import pipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from supabase.client import Client, create_client

load_dotenv()

# Tools
@tool
def multiply(a: int, b: int) -> int:
    return a * b

@tool
def add(a: int, b: int) -> int:
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    return a - b

@tool
def divide(a: int, b: int) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    return a % b

@tool
def wiki_search(query: str) -> str:
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    return "\n\n---\n\n".join([doc.page_content for doc in search_docs])

@tool
def web_search(query: str) -> str:
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    return "\n\n---\n\n".join([doc.page_content for doc in search_docs])

@tool
def arvix_search(query: str) -> str:
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    return "\n\n---\n\n".join([doc.page_content[:1000] for doc in search_docs])

tools = [multiply, add, subtract, divide, modulus, wiki_search, web_search, arvix_search]

# Load system prompt
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()
sys_msg = SystemMessage(content=system_prompt)

# Vector store setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_SERVICE_KEY")
)
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents_langchain"
)

# Mistral agent
class MistralAgent:
    def __init__(self):
        self.generator = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1", device=0)
        print("Mistral model loaded.")

    def invoke(self, messages):
        question = messages[-1].content
        result = self.generator(question, max_length=300, do_sample=True)[0]["generated_text"]
        return HumanMessage(content=result.strip())

mistral_agent = MistralAgent()

# LangGraph builder
def build_graph():
    def assistant(state: MessagesState):
        return {"messages": [mistral_agent.invoke(state["messages"])]}

    def retriever(state: MessagesState):
        similar = vector_store.similarity_search(state["messages"][-1].content)
        example = HumanMessage(content=f"Similar Q&A:\n\n{similar[0].page_content}")
        return {"messages": [sys_msg] + state["messages"] + [example]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()

# Run the agent
def run_agent(question: str) -> str:
    graph = build_graph()
    messages = [HumanMessage(content=question)]
    result = graph.invoke({"messages": messages})
    return result["messages"][-1].content.strip()

if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    answer = run_agent(question)
    print("ANSWER:", answer)
