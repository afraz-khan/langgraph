"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langchain_aws import ChatBedrock


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    aws_region: str
    temperature: float


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    messages: List[BaseMessage]


def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input messages and returns AI response using Bedrock Claude 3.5v2.

    Uses runtime context to configure the model.
    """
    # Get configuration from runtime context
    # aws_region = runtime.context.get("aws_region", "us-west-2")
    # temperature = runtime.context.get("temperature", 0.7)
    aws_region = "us-west-2"
    temperature = 0.7

    # Initialize the Bedrock Claude model
    model = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name=aws_region,
        model_kwargs={
            "temperature": temperature,
            "max_tokens": 4096,
        },
    )

    # Invoke the model with the conversation history
    response = model.invoke(state.messages)

    # Return the new message to be added to state
    return {"messages": [response]}


# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node("chat_model", call_model)
    .add_edge("__start__", "chat_model")
    .compile(name="Bedrock Claude Chat")
)
