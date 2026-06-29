from operator import add

from langchain_core.messages import AnyMessage
from typing_extensions import Annotated, TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    llm_calls: int
    rewrite_count: int
    retrieved_docs: list[str]
    last_tool_names: list[str]
    docs_relevant: bool
