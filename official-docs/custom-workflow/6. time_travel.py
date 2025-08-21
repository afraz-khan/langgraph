import json
from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage

memory = InMemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


graph_builder = StateGraph(State)


llm = init_chat_model(
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_provider="bedrock_converse",
)

# Create a separate LLM instance for parsing human responses
parser_llm = init_chat_model(
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_provider="bedrock_converse",
)


def parse_human_response(human_text: str) -> dict:
    """
    Parse natural language human response and extract name/birthday information.
    Returns a dictionary with extracted information and correctness confirmation.
    """

    prompt = f"""You are a helpful assistant that analyzes human responses to determine if they are confirming information is correct OR providing new/corrected information.

Human Response: "{human_text}"

Your task is to analyze the human response and determine:
1. Whether they are confirming that information is correct
2. OR if they are providing new name and birthday information

Please respond with ONLY a JSON object in this exact format:
{{
    "correct": true/false,
    "name": "extracted name or null if not provided",
    "birthday": "extracted birthday or null if not provided",
    "confidence": "high/medium/low",
    "reasoning": "brief explanation of your interpretation"
}}

Guidelines:

**For Confirmation (correct: true):**
- Set "correct" to true if they say things like: "yes", "correct", "that's right", "looks good", "ok", "fine", "accurate", etc.
- When correct=true, set name and birthday to null (since they're not providing new info)

**For New Information (correct: false):**
- Set "correct" to false if they provide any new information or corrections
- Extract name from patterns like: "my name is X", "I'm X", "call me X", "actually it's X"
- Extract birthday from any date mentioned: MM/DD/YYYY, Month Day Year, DD-MM-YYYY, YYYY-MM-DD, etc.
- Convert dates to YYYY-MM-DD format when possible
- Look for phrases like "born on", "birthday is", "I was born", etc.

Examples:
- "Yes, that's correct" 
  → {{"correct": true, "name": null, "birthday": null, "confidence": "high", "reasoning": "User confirmed information is correct"}}
  
- "Looks good to me"
  → {{"correct": true, "name": null, "birthday": null, "confidence": "high", "reasoning": "User confirmed information is correct"}}
  
- "My name is John Smith and I was born on March 15, 1990"
  → {{"correct": false, "name": "John Smith", "birthday": "1990-03-15", "confidence": "high", "reasoning": "User provided new name and birthday"}}
  
- "Actually, call me Mike"
  → {{"correct": false, "name": "Mike", "birthday": null, "confidence": "high", "reasoning": "User corrected name only"}}
  
- "No, my birthday is 12/25/1985"
  → {{"correct": false, "name": null, "birthday": "1985-12-25", "confidence": "high", "reasoning": "User corrected birthday only"}}

Be precise in determining confirmation vs. new information.
"""

    try:
        response = parser_llm.invoke(
            [
                SystemMessage(
                    content="You are a precise information extraction assistant. Always respond with valid JSON only."
                ),
                HumanMessage(content=prompt),
            ]
        )

        # Extract JSON from the response
        response_text = response.content.strip()

        # Handle cases where the response might have markdown code blocks
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        parsed_response = json.loads(response_text)

        # Validate the response has required fields
        required_fields = ["correct", "name", "birthday", "confidence", "reasoning"]
        if not all(field in parsed_response for field in required_fields):
            raise ValueError("Missing required fields in parsed response")

        return parsed_response

    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"Error parsing human response: {e}")
        print(
            f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}"
        )

        # Fallback: simple keyword-based parsing
        human_lower = human_text.lower().strip()

        # Check for simple confirmations
        if any(
            word in human_lower
            for word in [
                "yes",
                "correct",
                "right",
                "good",
                "ok",
                "okay",
                "fine",
                "accurate",
            ]
        ):
            return {
                "correct": True,
                "name": None,
                "birthday": None,
                "confidence": "medium",
                "reasoning": "Fallback: detected confirmation keywords",
            }
        else:
            return {
                "correct": False,
                "name": None,
                "birthday": None,
                "confidence": "low",
                "reasoning": f"Fallback: could not parse response. Error: {str(e)}",
            }


@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct") is True:
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)


tavily_tool = TavilySearch(max_results=2)
tools = [tavily_tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tavily_tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
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
        # Parse the human response when resuming from tools node
        parsed_response = parse_human_response(user_input)
        print(f"Parsed human response: {parsed_response}")

        # Wrap parsed response in Command if tools node is next
        final_user_input = Command(resume=parsed_response)

    for event in graph.stream(
        final_user_input,
        config=config,
        stream_mode="values",
    ):
        if "messages" in event:
            event["messages"][-1].pretty_print()


def stream_graph_updates_rewind(to_replay):
    for event in graph.stream(
        None,
        config=to_replay.config,
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
            print(snapshot)
            continue
        elif user_input.lower() == "rewind":
            to_replay = None
            for state in graph.get_state_history(config):
                print(
                    "Num Messages: ",
                    len(state.values["messages"]),
                    "Next: ",
                    state.next,
                )
                print("-" * 80)
                if len(state.values["messages"]) == 2:
                    # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
                    to_replay = state
                    print(to_replay.next)
                    print(to_replay.config)
                    stream_graph_updates_rewind(to_replay)
                    break
            continue

        stream_graph_updates(user_input)
    except Exception as e:
        print(f"Error: {e}")
        # fallback if input() is not available
        # user_input = "What do you know about LangGraph?"
        # print("User: " + user_input)
        # stream_graph_updates(user_input)
        break
