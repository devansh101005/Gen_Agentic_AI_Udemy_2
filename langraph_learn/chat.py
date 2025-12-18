from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain.chat_models import init_chat_model

load_dotenv()

llm=init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai"
)


class State(TypedDict):
    messages: Annotated[list,add_messages]


#This is a node which return below messages
def chatbot(state:State):
    #print("\n\n Inside chatbot node",state)
    response=llm.invoke(state.get("messages"))
    #return {"messages":["Hi,This is a message from ChatBot Node"]}
    return {"messages":[response]}

def samplenode(state:State):
    print("\n\nInside samplenode node",state)
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

updated_state=graph.invoke(State({"messages":["Hi,My name is Piyush Garg"]}))
print("\n\n updated_state",updated_state)