from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage,HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain

load_dotenv()

graph = MessageGraph()

REFLECT = 'reflect'
GENERATE = 'generate'

def generate_node(state):
    return generation_chain.invoke({
        "messages": state
    })

# def reflect_node(state):
#     response =  reflection_chain.invoke({
#         "messages": state
#     })
#     return [HumanMessage(content=response.content)]

def reflect_node(state):
    # Swap roles: treat the AI's last post as if a human submitted it for review,
    # so the final turn Gemini sees is a HumanMessage, not an AIMessage.
    translated = []
    for msg in state:
        if isinstance(msg, HumanMessage):
            translated.append(AIMessage(content=msg.content))
        else:
            translated.append(HumanMessage(content=msg.content))

    response = reflection_chain.invoke({"messages": translated})
    return [HumanMessage(content=response.content)]

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)

def should_continue(state):
    if(len(state) > 4):
        return END
    return REFLECT

graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke([HumanMessage(content="Write a LinkedIn post about the benefits of using AI in business.")])

print("Final Response:", response)