from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from langchain_tavily import TavilySearch

memory = InMemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = init_chat_model(
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_provider="bedrock_converse",
)


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


tavily_tool = TavilySearch(max_results=2)
tools = [tavily_tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges("chatbot", tools_condition)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile(checkpointer=memory)

try:
    # Generate the PNG bytes
    png_bytes = graph.get_graph().draw_mermaid_png()

    # Save to file
    with open("chatbot_graph.png", "wb") as f:
        f.write(png_bytes)

    print("Graph image saved...")
except Exception as e:
    # This requires some extra dependencies and is optional
    print(f"Could not generate graph: {e}")
    print(
        "You may need to install additional dependencies like 'pip install pygraphviz' or 'brew install graphviz'"
    )


config = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(user_input: str):
    final_user_input = {"messages": [{"role": "user", "content": user_input}]}

    # Check if tools node is next from snapshot
    snapshot = graph.get_state(config)
    if snapshot and "tools" in snapshot.next:
        # Wrap user input in Command if tools node is next
        final_user_input = Command(resume={"data": user_input})

    for event in graph.stream(
        final_user_input,
        config=config,
        stream_mode="values",
    ):
        if "messages" in event:
            event["messages"][-1].pretty_print()


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "snapshot":
            # Get current snapshot from memory
            snapshot = graph.get_state(config)
            print("\nCurrent conversation snapshot:")
            # for message in snapshot["messages"]:
            #     if hasattr(message, "pretty_print"):
            #         message.pretty_print()
            #     else:
            #         print(f"{message['role'].title()}: {message['content']}")
            print(snapshot)
            continue

        stream_graph_updates(user_input)
    except Exception as e:
        print(f"Error: {e}")
        # fallback if input() is not available
        # user_input = "What do you know about LangGraph?"
        # print("User: " + user_input)
        # stream_graph_updates(user_input)
        break
