from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END


class State(TypedDict):
    messages: Annotated[list,add_messages]


#This is a node which return below messages
def chatbot(state:State):
    print("Inside chatbot node",state)
    return {"messages":["Hi,This is a message from ChatBot Node"]}

def samplenode(state:State):
    print("Inside samplenode node",state)
    return {"messages":["Sample Message Appended"]}

graph_builder=StateGraph(State)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("samplenode",samplenode)
#A node is just a function that does tasks



# These are edges which are connecting nodes
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot","samplenode")
graph_builder.add_edge("samplenode",END)

graph =graph_builder.compile()