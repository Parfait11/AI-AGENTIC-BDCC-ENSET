import asyncio
import threading
import time
import socket
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

# Serveur MCP HTTP local (simulation d'un serveur distant)
mcp_server = FastMCP("travel_server", host=SERVER_HOST, port=SERVER_PORT)


@mcp_server.tool()
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights between two cities on a given date."""
    return (
        f"Flights from {origin} to {destination} on {date}:\n"
        f"- AT 601 | 08:00 → 09:15 | Direct | 850 MAD\n"
        f"- AT 603 | 14:30 → 15:45 | Direct | 920 MAD\n"
        f"- AT 605 | 19:00 → 20:15 | Direct | 780 MAD"
    )


@mcp_server.tool()
def get_flight_price(flight_number: str) -> str:
    """Get the price and details of a specific flight."""
    prices = {
        "AT 601": "850 MAD - Economy class, 1 bag included",
        "AT 603": "920 MAD - Economy class, 1 bag included",
        "AT 605": "780 MAD - Economy class, carry-on only",
    }
    return prices.get(flight_number, f"Flight {flight_number} not found")


def _wait_for_server(host: str, port: int, timeout: int = 10) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _start_server():
    mcp_server.run(transport="streamable-http")


async def main():
    # Démarrer le serveur HTTP MCP dans un thread daemon
    server_thread = threading.Thread(target=_start_server, daemon=True)
    server_thread.start()

    if not _wait_for_server(SERVER_HOST, SERVER_PORT):
        raise RuntimeError("MCP HTTP server did not start in time")

    print(f"Serveur MCP HTTP démarré sur http://{SERVER_HOST}:{SERVER_PORT}/mcp")

    client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": f"http://{SERVER_HOST}:{SERVER_PORT}/mcp"
            }
        }
    )

    # get tools
    tools = await client.get_tools()
    print(f"Tools disponibles : {[t.name for t in tools]}")

    model = ChatOllama(model="llama3.2:3b")

    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt="You are a travel agent. No follow up questions."
    )

    config = {"configurable": {"thread_id": "1"}}

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Get me a direct flight from Rabat to Agadir on August 31st")]},
        config
    )

    print(response['messages'][-1].content)


asyncio.run(main())
