from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from src.graph import route_after_agent, route_after_grading, route_after_tools


def test_route_after_agent_with_tool_calls_goes_to_tools():
    state = {
        "messages": [
            HumanMessage(content="Bonjour"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "retrieve_documents", "args": {"query": "budget"}, "id": "1"}
                ],
            ),
        ]
    }
    assert route_after_agent(state) == "tools"


def test_route_after_agent_without_tool_calls_ends():
    state = {
        "messages": [
            HumanMessage(content="Bonjour"),
            AIMessage(content="Voici la reponse."),
        ]
    }
    assert route_after_agent(state) == END


def test_route_after_tools_with_retrieval_goes_to_grading():
    state = {"last_tool_names": ["retrieve_documents"]}
    assert route_after_tools(state) == "grade_documents"


# def test_route_after_tools_without_retrieval_goes_to_agent():
#     state = {"last_tool_names": ["compute_loan_payment"]}
#     assert route_after_tools(state) == "agent"


def test_route_after_grading_relevant_goes_to_agent():
    state = {"docs_relevant": True, "rewrite_count": 0}
    assert route_after_grading(state) == "agent"


def test_route_after_grading_not_relevant_goes_to_rewrite():
    state = {"docs_relevant": False, "rewrite_count": 0}
    assert route_after_grading(state) == "rewrite_query"


def test_route_after_grading_not_relevant_but_max_rewrites_goes_to_agent():
    state = {"docs_relevant": False, "rewrite_count": 2}
    assert route_after_grading(state) == "agent"
