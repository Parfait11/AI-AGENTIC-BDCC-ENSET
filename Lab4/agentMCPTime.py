import asyncio
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama


async def main():
    client = MultiServerMCPClient(
        {
            "time": {
                "transport": "stdio",
                "command": "mcp-server-time",
                "args": [
                    "--local-timezone=America/New_York"
                ]
            }
        }
    )

    # get tools
    tools = await client.get_tools()

    # Initialiser le modèle Ollama
    model = ChatOllama(
        model="llama3.2:3b",
    )

    agent = create_agent(
        model=model,
        tools=tools,
    )

    question = HumanMessage(content="What time is it in Japan")

    response = await agent.ainvoke(
        {"messages": [question]}
    )

    print(response['messages'][-1].content)


asyncio.run(main())
