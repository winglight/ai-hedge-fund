import asyncio
import json
import re
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.backend.services.agent_service import create_agent_function
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.main import start
from src.utils.analysts import ANALYST_CONFIG
from src.graph.state import AgentState
from src.data.providers import DEFAULT_PROVIDER_NAME, provider_context

DATA_PROVIDER_SETTING = "DATA_PROVIDER"


def extract_base_agent_key(unique_id: str) -> str:
    """
    Extract the base agent key from a unique node ID.
    
    Args:
        unique_id: The unique node ID with suffix (e.g., "warren_buffett_abc123")
    
    Returns:
        The base agent key (e.g., "warren_buffett")
    """
    # For agent nodes, remove the last underscore and 6-character suffix
    parts = unique_id.split('_')
    if len(parts) >= 2:
        last_part = parts[-1]
        # If the last part is a 6-character alphanumeric string, it's likely our suffix
        if len(last_part) == 6 and re.match(r'^[a-z0-9]+$', last_part):
            return '_'.join(parts[:-1])
    return unique_id  # Return original if no suffix pattern found


# Helper function to create the agent graph
def create_graph(graph_nodes: list, graph_edges: list) -> StateGraph:
    """Create the workflow based on the React Flow graph structure."""
    graph = StateGraph(AgentState)
    graph.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = {key: (f"{key}_agent", config["agent_func"]) for key, config in ANALYST_CONFIG.items()}
    
    # Extract agent IDs from graph structure
    agent_ids = [node.id for node in graph_nodes]
    agent_ids_set = set(agent_ids)
    
    # Track which nodes are portfolio managers for special handling
    portfolio_manager_nodes = set()
    
    # Add agent nodes
    for unique_agent_id in agent_ids:
        base_agent_key = extract_base_agent_key(unique_agent_id)
        
        # Track portfolio manager nodes for special handling (before ANALYST_CONFIG check)
        if base_agent_key == "portfolio_manager":
            portfolio_manager_nodes.add(unique_agent_id)
            continue
            
        # Skip if the base agent key is not in our analyst configuration
        if base_agent_key not in ANALYST_CONFIG:
            continue
            
        node_name, node_func = analyst_nodes[base_agent_key]
        agent_function = create_agent_function(node_func, unique_agent_id)
        graph.add_node(unique_agent_id, agent_function)
    
    # Add portfolio manager nodes and their corresponding risk managers
    risk_manager_nodes = {}  # Map portfolio manager ID to risk manager ID
    for portfolio_manager_id in portfolio_manager_nodes:
        portfolio_manager_function = create_agent_function(portfolio_management_agent, portfolio_manager_id)
        graph.add_node(portfolio_manager_id, portfolio_manager_function)
        
        # Create unique risk manager for this portfolio manager
        suffix = portfolio_manager_id.split('_')[-1]
        risk_manager_id = f"risk_management_agent_{suffix}"
        risk_manager_nodes[portfolio_manager_id] = risk_manager_id
        
        # Add the risk manager node
        risk_manager_function = create_agent_function(risk_management_agent, risk_manager_id)
        graph.add_node(risk_manager_id, risk_manager_function)

    # Build connections based on React Flow graph structure
    nodes_with_incoming_edges = set()
    nodes_with_outgoing_edges = set()
    direct_to_portfolio_managers = {}  # Map analyst ID to portfolio manager ID for direct connections
    
    for edge in graph_edges:
        # Only consider edges between agent nodes (not from stock tickers)
        if edge.source in agent_ids_set and edge.target in agent_ids_set:
            source_base_key = extract_base_agent_key(edge.source)
            target_base_key = extract_base_agent_key(edge.target)
            
            nodes_with_incoming_edges.add(edge.target)
            nodes_with_outgoing_edges.add(edge.source)
            
            # Check if this is a direct connection from analyst to portfolio manager
            if (source_base_key in ANALYST_CONFIG and 
                source_base_key != "portfolio_manager" and 
                target_base_key == "portfolio_manager"):
                # Don't add direct edge to portfolio manager - we'll route through risk manager
                direct_to_portfolio_managers[edge.source] = edge.target
            else:
                # Add edge between agent nodes (but not direct to portfolio managers)
                graph.add_edge(edge.source, edge.target)
    
    # Connect start_node to nodes that don't have incoming edges from other agents
    for agent_id in agent_ids:
        if agent_id not in nodes_with_incoming_edges:
            base_agent_key = extract_base_agent_key(agent_id)
            if base_agent_key in ANALYST_CONFIG and base_agent_key != "portfolio_manager":
                graph.add_edge("start_node", agent_id)
    
    # Connect analysts that have direct connections to portfolio managers to their corresponding risk managers
    for analyst_id, portfolio_manager_id in direct_to_portfolio_managers.items():
        risk_manager_id = risk_manager_nodes[portfolio_manager_id]
        graph.add_edge(analyst_id, risk_manager_id)
    
    # Connect each risk manager to its corresponding portfolio manager
    for portfolio_manager_id, risk_manager_id in risk_manager_nodes.items():
        graph.add_edge(risk_manager_id, portfolio_manager_id)
    
    # Connect portfolio managers to END
    for portfolio_manager_id in portfolio_manager_nodes:
        graph.add_edge(portfolio_manager_id, END)

    # Set the entry point to the start node
    graph.set_entry_point("start_node")
    return graph


async def run_graph_async(graph, portfolio, tickers, start_date, end_date, model_name, model_provider, request=None):
    """Async wrapper for run_graph to work with asyncio."""
    # Use run_in_executor to run the synchronous function in a separate thread
    # so it doesn't block the event loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: run_graph(graph, portfolio, tickers, start_date, end_date, model_name, model_provider, request))  # Use default executor
    return result


def run_graph(
    graph: StateGraph,
    portfolio: dict,
    tickers: list[str],
    start_date: str,
    end_date: str,
    model_name: str,
    model_provider: str,
    request=None,
) -> dict:
    """
    Run the graph with the given portfolio, tickers,
    start date, end date, show reasoning, model name,
    and model provider.
    """
    provider_name, credentials = build_provider_credentials(request)
    strategy_mode = getattr(request, "strategy_mode", None) if request else None
    data_timeframe = getattr(request, "data_timeframe", None) if request else None
    data_granularity = (
        "intraday"
        if (strategy_mode and "intra" in strategy_mode.lower())
        or (data_timeframe and any(ch in str(data_timeframe).lower() for ch in ["m", "min", "hour", "hr", "h"]))
        else "end_of_day"
    )
    with provider_context(provider_name, credentials):
        return graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                    "workflow_metadata": {
                        "strategy_mode": strategy_mode,
                        "data_timeframe": data_timeframe,
                        "data_provider": provider_name,
                        "data_granularity": data_granularity,
                    },
                },
                "metadata": {
                    "show_reasoning": False,
                    "model_name": model_name,
                    "model_provider": model_provider,
                    "request": request,  # Pass the request for agent-specific model access
                    "data_provider": provider_name,
                    "strategy_mode": strategy_mode,
                    "data_timeframe": data_timeframe,
                    "data_granularity": data_granularity,
                },
            },
        )


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None
# Provider utilities -------------------------------------------------------


def build_provider_credentials(request) -> tuple[str, dict]:
    if request is None:
        return DEFAULT_PROVIDER_NAME, {}

    api_keys = getattr(request, "api_keys", {}) or {}
    provider_name = getattr(request, "data_provider", None) or api_keys.get(DATA_PROVIDER_SETTING) or DEFAULT_PROVIDER_NAME
    provider_options = dict(getattr(request, "data_provider_options", {}) or {})

    credentials = dict(provider_options)

    if provider_name == "financial_datasets":
        api_key = api_keys.get("FINANCIAL_DATASETS_API_KEY")
        if api_key:
            credentials.setdefault("api_key", api_key)
    elif provider_name == "ibbot":
        mapping = {
            "host": "IBBOT_HOST",
            "account": "IBBOT_ACCOUNT",
            "access_token": "IBBOT_ACCESS_TOKEN",
            "refresh_token": "IBBOT_REFRESH_TOKEN",
        }
        for field, key_name in mapping.items():
            value = api_keys.get(key_name)
            if value:
                credentials.setdefault(field, value)

    return provider_name, credentials

