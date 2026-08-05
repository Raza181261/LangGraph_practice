from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

sqlite_conn = sqlite3.connect("chat_history.db", check_same_thread=False)

memory = SqliteSaver(sqlite_conn)

llm = ChatGroq(model="llama-3.1-8b-instant")

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: BasicChatState):
    return{
        'messages': [llm.invoke(state['messages'])]
    }


graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=memory)

cofig = {"configurable": {
    "thread_id": 1
}}

while True:
    user_input = input("You: ")
    if (user_input in ["exit", "quit", "end"]):
        break
    else:
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
            }, config=cofig)
        print("AI: " + result["messages"][-1].content)